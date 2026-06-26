"""
工具注册表 — 基于 ToolAdapter 统一管理所有 Agent 工具。
新增工具只需 3 行代码：定义 handler → 创建 ToolAdapter → register()。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from knowledge.base import KnowledgeBase
from memory.long_term import LongTermMemory
from tools.adapter import ToolAdapter, ToolMeta

logger = logging.getLogger(__name__)


class ToolRegistry:

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        tool_timeout_seconds: float = 30.0,
    ):
        self._long_term_memory = long_term_memory
        self._kb = KnowledgeBase(long_term_memory)
        self._adapters: dict[str, ToolAdapter] = {}
        self._notifications: list[dict] = []
        self._tool_timeout = tool_timeout_seconds

        self._build_default_adapters()

    def _build_default_adapters(self):
        registry = self

        async def _knowledge_search_handler(query: str) -> str:
            try:
                results = await asyncio.wait_for(
                    registry._kb.search(query, top_k=3),
                    timeout=registry._tool_timeout,
                )
                return json.dumps(
                    results or [{"content": f"未找到关于'{query}'的相关知识", "source": "N/A", "score": 0}],
                    ensure_ascii=False, indent=2,
                )
            except asyncio.TimeoutError:
                logger.warning("knowledge_search 超时: query=%r", query)
                return json.dumps([{"content": "检索超时，请稍后重试", "source": "N/A", "score": 0}])
            except Exception as e:
                logger.error("knowledge_search 失败: %s", e)
                return json.dumps([{"content": f"检索失败: {e}", "source": "N/A", "score": 0}])

        self._adapters["knowledge_search"] = ToolAdapter(
            meta=ToolMeta(
                name="knowledge_search",
                description="搜索企业知识库。当需要查找产品信息、政策条款、操作流程等知识时使用此工具。",
                category="knowledge",
                timeout_seconds=self._tool_timeout,
            ),
            handler=_knowledge_search_handler,
        )

        self._adapters["ticket_create"] = ToolAdapter(
            meta=ToolMeta(
                name="ticket_create",
                description="创建客服工单。当用户需要退款、理赔、投诉、开户等需要人工处理的业务时使用。",
                category="ticket",
                risk_level="medium",
            ),
            handler=lambda user_id, description, priority="medium": json.dumps(
                {"ticket_id": f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                 "status": "created", "priority": priority},
                ensure_ascii=False,
            ),
        )

        self._adapters["order_query"] = ToolAdapter(
            meta=ToolMeta(
                name="order_query",
                description="查询订单或工单信息。可通过订单ID或用户ID查询。",
                category="ticket",
            ),
            handler=lambda order_id="", user_id="": json.dumps(
                {"order_id": order_id or "ORD-20260401-001", "status": "shipped", "amount": 299.00},
                ensure_ascii=False,
            ),
        )

        self._adapters["risk_check"] = ToolAdapter(
            meta=ToolMeta(
                name="risk_check",
                description="风控检查。在执行敏感操作（转账、提现、退款等）前必须调用此工具进行风险评估。",
                category="risk",
                risk_level="high",
            ),
            handler=self._risk_check_handler,
        )

        self._adapters["notification_send"] = ToolAdapter(
            meta=ToolMeta(
                name="notification_send",
                description="发送通知给用户。工单创建成功后通知用户处理进度。",
                category="notification",
            ),
            handler=self._notification_send_handler,
        )

    @staticmethod
    def _risk_check_handler(action: str, amount: float = 0.0) -> str:
        risk_level = "low"
        warnings = []
        action_lower = action.lower()
        sensitive = ["转账", "提现", "修改密码", "绑定", "解绑", "注销", "transfer", "withdraw", "password"]
        if any(kw in action_lower for kw in sensitive):
            risk_level = "medium"
            warnings.append("敏感操作")
        if amount > 50000:
            risk_level = "high"
            warnings.append(f"大额交易: {amount}元")
        return json.dumps({
            "risk_level": risk_level,
            "warnings": warnings,
            "requires_manual_review": risk_level == "high",
        }, ensure_ascii=False)

    def _notification_send_handler(self, user_id: str, ticket_id: str, channel: str = "sms") -> str:
        notification = {
            "id": uuid.uuid4().hex[:8],
            "user_id": user_id,
            "ticket_id": ticket_id,
            "channel": channel,
            "message": f"工单 {ticket_id} 已创建，客服将尽快处理。",
            "sent_at": datetime.now().isoformat(),
            "status": "sent",
        }
        self._notifications.append(notification)
        return json.dumps(notification, ensure_ascii=False)

    def register(self, adapter: ToolAdapter) -> None:
        self._adapters[adapter.meta.name] = adapter

    def get_openai_functions(self) -> list[dict]:
        return [a.to_openai_function() for a in self._adapters.values()]

    def get_tool_specs(self) -> list[dict]:
        return [a.to_mcp_spec() for a in self._adapters.values()]

    def register_to_mcp(self, mcp_server) -> int:
        count = 0
        for adapter in self._adapters.values():
            mcp_server.register_tool(
                name=adapter.meta.name,
                description=adapter.meta.description,
                handler=adapter._handler,
            )
            count += 1
        return count

    async def seed_knowledge_base(self) -> int:
        return await self._kb.seed()

    def get_notifications(self) -> list[dict]:
        return self._notifications

    def get_tool_meta(self, name: str) -> ToolMeta | None:
        adapter = self._adapters.get(name)
        return adapter.meta if adapter else None

    def list_tool_metas(self) -> list[ToolMeta]:
        return [a.meta for a in self._adapters.values()]