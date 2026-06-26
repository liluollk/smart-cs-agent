"""
Agent 基类 — 通过 MCP Client 直接调用工具，增加短期记忆注入和重试机制。
- 工具调用：Agent → MCP Client → JSON-RPC → MCP Server → handler
- 短期记忆：Agent 在构建 prompt 时可获取多轮对话历史
- 重试：LLM 调用和工具调用支持指数退避重试
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from memory.short_term import ShortTermMemory
from tracing.collector import get_collector

logger = logging.getLogger(__name__)

MAX_LLM_RETRIES = 3
MAX_TOOL_RETRIES = 2
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 5.0


class ToolCallingAgent:

    def __init__(
            self,
            llm: ChatOpenAI,
            mcp_client: Any,
            openai_functions: list[dict],
            system_prompt: str,
            output_key: str,
            short_term_memory: ShortTermMemory | None = None,
    ):
        self.llm = llm
        self.mcp_client = mcp_client
        self.openai_functions = openai_functions
        self.system_prompt = system_prompt
        self.output_key = output_key
        self.short_term_memory = short_term_memory
        self.llm_with_tools = llm.bind(tools=openai_functions)

    async def _execute_tool(self, tool_name: str, args: dict[str, Any],
                            trace_id: str = "", session_id: str = "") -> str:
        collector = get_collector()
        start = time.time()

        for attempt in range(MAX_TOOL_RETRIES):
            try:
                result = await self.mcp_client.call_tool(tool_name, args)
                duration_ms = (time.time() - start) * 1000
                collector.record_tool_call(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    agent_name=self.output_key,
                    session_id=session_id,
                    success=True,
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )
                return str(result)
            except Exception as e:
                logger.warning("工具调用 %s 失败 (attempt=%d): %s", tool_name, attempt + 1, e)
                if attempt < MAX_TOOL_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                else:
                    duration_ms = (time.time() - start) * 1000
                    collector.record_tool_call(
                        tool_name=tool_name,
                        trace_id=trace_id,
                        agent_name=self.output_key,
                        session_id=session_id,
                        success=False,
                        duration_ms=duration_ms,
                        error=str(e),
                        retry_count=attempt + 1,
                    )
                    return json.dumps({"error": str(e)})

        duration_ms = (time.time() - start) * 1000
        collector.record_tool_call(
            tool_name=tool_name,
            trace_id=trace_id,
            agent_name=self.output_key,
            session_id=session_id,
            success=False,
            duration_ms=duration_ms,
            error="max retries exceeded",
            retry_count=MAX_TOOL_RETRIES,
        )
        return json.dumps({"error": "max retries exceeded"})

    async def _invoke_llm(self, messages: list[Any]) -> Any:
        last_error = None
        for attempt in range(MAX_LLM_RETRIES):
            try:
                return await self.llm_with_tools.ainvoke(messages)
            except Exception as e:
                last_error = e
                logger.warning("LLM 调用失败 (attempt=%d): %s", attempt + 1, e)
                if attempt < MAX_LLM_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
        raise last_error or RuntimeError("LLM 调用失败")

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        collector = get_collector()
        trace_id = state.get("trace_id", "")
        session_id = state.get("session_id", "")
        span = collector.start_span(self.output_key, trace_id=trace_id, method="process")

        messages = state.get("messages", [])
        if not messages:
            collector.end_span(span)
            return state

        try:
            system_msg = SystemMessage(content=self._build_system_prompt(state))
            user_msg = HumanMessage(content=await self._build_user_message(state))

            chat_history = [system_msg, user_msg]
            llm_result = await self._invoke_llm(chat_history)
            chat_history.append(llm_result)

            token_usage = (llm_result.response_metadata or {}).get("token_usage", {})
            if token_usage:
                collector.add_token_usage(span.span_id, token_usage)

            max_rounds = 5
            for _ in range(max_rounds):
                if not (hasattr(llm_result, "tool_calls") and llm_result.tool_calls):
                    break

                for tc in llm_result.tool_calls:
                    result_content = await self._execute_tool(
                        tc["name"], tc.get("args", {}),
                        trace_id=trace_id, session_id=session_id,
                    )
                    chat_history.append(
                        ToolMessage(content=result_content, tool_call_id=tc["id"])
                    )

                llm_result = await self._invoke_llm(chat_history)
                chat_history.append(llm_result)

                token_usage = (llm_result.response_metadata or {}).get("token_usage", {})
                if token_usage:
                    collector.add_token_usage(span.span_id, token_usage)

            answer = llm_result.content or ""

        except Exception as e:
            logger.error("Agent %s 处理失败: %s", self.output_key, e, exc_info=True)
            answer = self._fallback_message()

        collector.end_span(span)
        return {
            **state,
            "sub_results": {**state.get("sub_results", {}), self.output_key: answer},
        }

    def _build_system_prompt(self, _state: dict[str, Any]) -> str:
        return self.system_prompt

    async def _build_user_message(self, state: dict[str, Any]) -> str:
        messages = state.get("messages", [])
        current = messages[-1].content if messages else ""

        session_id = state.get("session_id", "")
        if self.short_term_memory and session_id:
            try:
                history = await self.short_term_memory.get_recent_messages(session_id, last_n=6)
                if history.strip():
                    return f"对话历史:\n{history}\n\n当前问题:\n<user_query>{current}</user_query>"
            except Exception as e:
                logger.warning("获取短期记忆失败: %s", e)

        return f"<user_query>{current}</user_query>"

    @staticmethod
    def _fallback_message() -> str:
        return "<fallback_message>抱歉，服务暂时不可用，请稍后重试或联系人工客服。>"