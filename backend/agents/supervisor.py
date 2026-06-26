"""
Supervisor — 多 Agent 编排器。
使用 AgentRegistry 动态发现 Agent，支持合规前置/后置检查分离。
Agent 通过 MCP Client 直接调用工具。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from core.registry import AgentRegistry
from memory.short_term import ShortTermMemory
from memory.working_memory import WorkingMemory
from tracing.collector import get_collector

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    trace_id: str
    intent: NotRequired[str]
    sub_results: NotRequired[dict[str, Any]]
    compliance_passed: NotRequired[bool]
    input_compliance_blocked: NotRequired[bool]
    input_compliance_violations: NotRequired[list[str]]
    final_response: NotRequired[str]
    current_agent: NotRequired[str]
    retry_count: NotRequired[int]
    needs_clarification: NotRequired[bool]


class Supervisor:

    def __init__(
            self,
            llm: ChatOpenAI,
            mcp_client: Any,
            agent_registry: AgentRegistry,
            router: Any,
            compliance_agent: Any,
            working_memory: WorkingMemory | None = None,
            short_term_memory: ShortTermMemory | None = None,
    ):
        self.llm = llm
        self.mcp_client = mcp_client
        self.working_memory = working_memory or WorkingMemory()
        self.short_term_memory = short_term_memory or ShortTermMemory()

        self.router = router
        self.compliance_agent = compliance_agent
        self._agent_registry = agent_registry

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    async def route_intent(self, state: AgentState) -> AgentState:
        collector = get_collector()
        trace_id = state.get("trace_id") or uuid.uuid4().hex[:12]
        span = collector.start_span("router", trace_id=trace_id)

        messages = state["messages"]
        user_message = messages[-1].content if messages else ""
        session_id = state.get("session_id", "")

        await self.short_term_memory.add_message(session_id, "user", user_message)

        wm_context = await self.working_memory.get_context(session_id)
        last_intent = wm_context.get("last_intent")
        last_agents = wm_context.get("last_agents", [])

        try:
            intent = await self.router.route(user_message, last_intent=last_intent, last_agents=last_agents, span_id=span.span_id)
            collector.end_span(span)
        except Exception as e:
            logger.error("意图路由失败: %s", e)
            intent = "knowledge"
            collector.end_span(span, error="意图路由失败")

        await self.working_memory.update(session_id, {
            "last_intent": intent,
            "last_user_message": user_message,
        })

        return dict(trace_id=trace_id, intent=intent, current_agent="router")  # type: ignore[return-value]

    @staticmethod
    def check_input_compliance(state: AgentState) -> str:
        if state.get("input_compliance_blocked", False):
            return "synthesize"
        return "fan_out"

    async def fan_out(self, state: AgentState) -> AgentState:
        collector = get_collector()
        trace_id = state.get("trace_id", "")
        span = collector.start_span("fan_out", trace_id=trace_id)

        intent = state.get("intent", "knowledge")
        session_id = state.get("session_id", "")

        agents = self._agent_registry.get_agents_for_intent(intent)
        tasks = [agent.process(state) for _, agent in agents]
        invoked_agents = [name for name, _ in agents]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = {**state}
        merged_sub = dict(state.get("sub_results", {}))
        for r in results:
            if isinstance(r, Exception):
                logger.error("Agent 执行异常: %s", r)
                continue
            if isinstance(r, dict):
                merged_sub.update(r.get("sub_results", {}))
                if r.get("needs_clarification"):
                    merged["needs_clarification"] = True
                if r.get("retry_count", 0) > merged.get("retry_count", 0):
                    merged["retry_count"] = r["retry_count"]

        await self.working_memory.update(session_id, {
            "last_agents": invoked_agents,
            "last_sub_results_keys": list(merged_sub.keys()),
        })

        merged["sub_results"] = merged_sub
        merged["current_agent"] = "fan_out"
        collector.end_span(span)
        return merged  # type: ignore[return-value]

    @staticmethod
    def check_clarification(state: AgentState) -> str:
        if state.get("needs_clarification", False):
            return "synthesize"
        return "compliance_post"

    async def synthesize(self, state: AgentState) -> AgentState:
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)
        needs_clarification = state.get("needs_clarification", False)
        input_blocked = state.get("input_compliance_blocked", False)
        session_id = state.get("session_id", "")

        if input_blocked:
            final_response = "您的消息包含敏感内容，已转交人工客服处理。"
            await self.short_term_memory.add_message(session_id, "assistant", final_response)
            await self.working_memory.update(session_id, {"last_response_type": "input_blocked"})
            return {  # type: ignore[return-value]
                **state,
                "final_response": final_response,
                "messages": [AIMessage(content=final_response)],
            }

        if needs_clarification:
            clarification = sub_results.get("ticket", "请提供更多信息。")
            await self.short_term_memory.add_message(session_id, "assistant", clarification)
            await self.working_memory.update(session_id, {"last_response_type": "clarification"})
            return {  # type: ignore[return-value]
                **state,
                "final_response": clarification,
                "messages": [AIMessage(content=clarification)],
            }

        if not compliance_passed:
            final_response = "您的请求涉及敏感内容，已转交人工客服处理。"
            await self.short_term_memory.add_message(session_id, "assistant", final_response)
            await self.working_memory.update(session_id, {"last_response_type": "compliance_blocked"})
        else:
            parts = [v for k, v in sub_results.items() if k != "compliance" and v]
            final_response = "\n\n".join(parts) if parts else "抱歉，暂时无法处理您的请求。"
            await self.short_term_memory.add_message(session_id, "assistant", final_response)
            await self.working_memory.update(session_id, {"last_response_type": "normal"})

        return {  # type: ignore[return-value]
            **state,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }

    def build_graph(self, checkpointer=None):
        graph = StateGraph(AgentState)  # type: ignore[arg-type]

        graph.add_node("router", self.route_intent)  # type: ignore[arg-type]
        graph.add_node("compliance_pre", self.compliance_agent.pre_check)  # type: ignore[arg-type]
        graph.add_node("fan_out", self.fan_out)  # type: ignore[arg-type]
        graph.add_node("compliance_post", self.compliance_agent.process)  # type: ignore[arg-type]
        graph.add_node("synthesize", self.synthesize)  # type: ignore[arg-type]

        graph.set_entry_point("router")
        graph.add_edge("router", "compliance_pre")

        graph.add_conditional_edges(
            "compliance_pre",
            self.check_input_compliance,
            {
                "fan_out": "fan_out",
                "synthesize": "synthesize",
            },
        )

        graph.add_conditional_edges(
            "fan_out",
            self.check_clarification,
            {
                "compliance_post": "compliance_post",
                "synthesize": "synthesize",
            },
        )

        graph.add_edge("compliance_post", "synthesize")
        graph.add_edge("synthesize", END)

        if checkpointer is None:
            checkpointer = InMemorySaver()
        return graph.compile(checkpointer=checkpointer)