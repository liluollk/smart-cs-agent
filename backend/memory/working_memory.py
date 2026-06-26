"""
工作记忆 — Agent 当前任务的中间推理状态。
使用共享 RedisBackend，按 session_id 隔离。
用于维护 Supervisor 的路由决策上下文和子 Agent 的中间结果。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from memory.redis_backend import RedisBackend


class WorkingMemory:

    def __init__(self, redis_url: str | None = None, max_entries_per_session: int = 50,
                 ttl_seconds: int = 1800):
        self._backend = RedisBackend(redis_url=redis_url)
        self._max_entries = max_entries_per_session
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"smartcs:wm:{session_id}"

    @staticmethod
    def _ctx_key(session_id: str) -> str:
        return f"smartcs:wm:ctx:{session_id}"

    async def update(self, session_id: str, data: dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "data": data}

        ctx_dict = {k: json.dumps(v, ensure_ascii=False) for k, v in data.items()}

        pipe = await self._backend.pipeline
        await pipe.rpush(self._session_key(session_id), json.dumps(entry, ensure_ascii=False))
        await pipe.ltrim(self._session_key(session_id), -self._max_entries, -1)
        await pipe.expire(self._session_key(session_id), self._ttl_seconds)
        if ctx_dict:
            await pipe.hset(self._ctx_key(session_id), ctx_dict)
            await pipe.expire(self._ctx_key(session_id), self._ttl_seconds)
        await pipe.execute()

    async def get_context(self, session_id: str) -> dict[str, Any]:
        raw = await self._backend.hgetall(self._ctx_key(session_id))
        return {k: json.loads(v) for k, v in raw.items()}

    async def get_history(self, session_id: str, last_n: int = 10) -> list[dict]:
        raw = await self._backend.lrange(self._session_key(session_id), -last_n, -1)
        return [json.loads(item) for item in raw]

    async def get_last_intent(self, session_id: str) -> str | None:
        ctx = await self.get_context(session_id)
        return ctx.get("last_intent")

    async def get_last_agents(self, session_id: str) -> list[str]:
        ctx = await self.get_context(session_id)
        return ctx.get("last_agents", [])

    async def get_turn_count(self, session_id: str) -> int:
        return await self._backend.llen(self._session_key(session_id))

    async def clear(self, session_id: str) -> None:
        await self._backend.delete(self._session_key(session_id), self._ctx_key(session_id))

    async def export_for_persistence(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "context": await self.get_context(session_id),
            "history": await self.get_history(session_id),
            "exported_at": datetime.now().isoformat(),
        }
