"""
短期记忆 — 基于Redis的会话级记忆。
存储最近N轮对话上下文，设置TTL自动过期。
使用共享 RedisBackend，消除重复代码。
"""

from __future__ import annotations

import json
from datetime import datetime

from memory.redis_backend import RedisBackend


class ShortTermMemory:

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_turns: int = 20,
        ttl_seconds: int = 1800,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._backend = RedisBackend(redis_url=redis_url)

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"smartcs:short_term:{session_id}"

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        key = self._session_key(session_id)
        await self._backend.rpush(key, json.dumps(message, ensure_ascii=False))
        await self._backend.ltrim(key, -self.max_turns, -1)
        await self._backend.expire(key, self.ttl_seconds)

    async def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        key = self._session_key(session_id)
        n = last_n or self.max_turns
        raw = await self._backend.lrange(key, -n, -1)
        return [json.loads(item) for item in raw]

    async def clear(self, session_id: str) -> None:
        await self._backend.delete(self._session_key(session_id))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_count = len(text) - cjk_count
        return int(cjk_count * 0.6 + other_count * 0.25) + 1

    async def get_context_window(self, session_id: str, max_tokens: int = 4000) -> str:
        history = await self.get_history(session_id)
        context_parts = []
        estimated_tokens = 0

        for msg in reversed(history):
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_tokens = self._estimate_tokens(msg_text)
            if estimated_tokens + msg_tokens > max_tokens:
                break
            context_parts.insert(0, msg_text)
            estimated_tokens += msg_tokens

        return "\n".join(context_parts)

    async def get_recent_messages(self, session_id: str, last_n: int = 5) -> str:
        history = await self.get_history(session_id, last_n=last_n)
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)