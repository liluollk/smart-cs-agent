"""
共享 Redis 后端 — 提供统一的 Redis 连接管理和 fallback 机制。
消除 WorkingMemory 和 ShortTermMemory 中的重复代码。
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None


class RedisBackend:
    """统一的 Redis 客户端封装，支持连接池、fallback 到进程内存。"""

    def __init__(self, redis_url: str | None = None, max_connections: int = 20):
        self._redis_url = redis_url
        self._max_connections = max_connections
        self._redis: Any = None
        self._pool: Any = None
        self._fallback: dict[str, Any] = defaultdict(dict)
        self._lock = threading.Lock()

    @property
    def redis_url(self) -> str | None:
        return self._redis_url

    async def get_client(self) -> Any | None:
        if self._redis is not None:
            return self._redis

        if not self._redis_url:
            return None

        if aioredis is None:
            return None

        try:
            if self._pool is None:
                self._pool = aioredis.ConnectionPool.from_url(
                    self._redis_url,
                    decode_responses=True,
                    max_connections=self._max_connections,
                )
            self._redis = aioredis.Redis(connection_pool=self._pool)
            await self._redis.ping()
        except Exception:
            self._redis = None
            self._redis_url = None
            return None

        return self._redis

    def is_available(self) -> bool:
        return self._redis is not None

    @staticmethod
    def _fallback_key(*parts: str) -> str:
        return ":".join(parts)

    # ---- Redis 操作（带 fallback） ----

    async def rpush(self, key: str, *values: str) -> int:
        r = await self.get_client()
        if r is not None:
            return await r.rpush(key, *values)
        with self._lock:
            fb = self._fallback_key(key)
            lst = self._fallback.get(fb, [])
            lst.extend(values)
            self._fallback[fb] = lst
            return len(lst)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        r = await self.get_client()
        if r is not None:
            return await r.lrange(key, start, end)
        with self._lock:
            fb = self._fallback_key(key)
            lst = self._fallback.get(fb, [])
            return lst[max(0, start):end if end >= 0 else None]

    async def ltrim(self, key: str, start: int, end: int) -> None:
        r = await self.get_client()
        if r is not None:
            await r.ltrim(key, start, end)
            return
        with self._lock:
            fb = self._fallback_key(key)
            lst = self._fallback.get(fb, [])
            self._fallback[fb] = lst[max(0, start):end + 1 if end >= 0 else None]

    async def llen(self, key: str) -> int:
        r = await self.get_client()
        if r is not None:
            return await r.llen(key)
        with self._lock:
            fb = self._fallback_key(key)
            return len(self._fallback.get(fb, []))

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        r = await self.get_client()
        if r is not None:
            return await r.hset(key, mapping=mapping)
        with self._lock:
            fb = self._fallback_key(key)
            existing = self._fallback.get(fb, {})
            existing.update(mapping)
            self._fallback[fb] = existing
            return len(mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        r = await self.get_client()
        if r is not None:
            return await r.hgetall(key)
        with self._lock:
            fb = self._fallback_key(key)
            return dict(self._fallback.get(fb, {}))

    async def expire(self, key: str, seconds: int) -> bool:
        r = await self.get_client()
        if r is not None:
            return await r.expire(key, seconds)
        return True

    async def delete(self, *keys: str) -> int:
        r = await self.get_client()
        if r is not None:
            return await r.delete(*keys)
        count = 0
        with self._lock:
            for key in keys:
                fb = self._fallback_key(key)
                if fb in self._fallback:
                    del self._fallback[fb]
                    count += 1
        return count

    @property
    async def pipeline(self) -> _FallbackPipeline:
        r = await self.get_client()
        if r is not None:
            return _FallbackPipeline(await r.pipeline())
        return _FallbackPipeline(self)


class _Pipeline:
    """Pipeline 抽象基类"""

    async def rpush(self, key: str, *values: str): ...
    async def ltrim(self, key: str, start: int, end: int): ...
    async def expire(self, key: str, seconds: int): ...
    async def hset(self, key: str, mapping: dict[str, str]): ...
    async def execute(self): ...


class _RedisPipeline:
    def __init__(self, pipe):
        self._pipe = pipe

    async def rpush(self, key: str, *values: str):
        self._pipe.rpush(key, *values)

    async def ltrim(self, key: str, start: int, end: int):
        self._pipe.ltrim(key, start, end)

    async def expire(self, key: str, seconds: int):
        self._pipe.expire(key, seconds)

    async def hset(self, key: str, mapping: dict[str, str]):
        self._pipe.hset(key, mapping=mapping)

    async def execute(self):
        await self._pipe.execute()


class _FallbackPipeline:
    def __init__(self, backend: RedisBackend):
        self._backend = backend
        self._ops: list[tuple[str, tuple]] = []

    async def rpush(self, key: str, *values: str):
        self._ops.append(("rpush", (key,) + values))

    async def ltrim(self, key: str, start: int, end: int):
        self._ops.append(("ltrim", (key, start, end)))

    async def expire(self, key: str, seconds: int):
        self._ops.append(("expire", (key, seconds)))

    async def hset(self, key: str, mapping: dict[str, str]):
        self._ops.append(("hset", (key, mapping)))

    async def execute(self):
        for op_name, args in self._ops:
            if op_name == "rpush":
                await self._backend.rpush(*args)
            elif op_name == "ltrim":
                await self._backend.ltrim(*args)
            elif op_name == "expire":
                await self._backend.expire(*args)
            elif op_name == "hset":
                await self._backend.hset(*args)