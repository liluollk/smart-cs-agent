"""
知识库 — 统一知识检索接口。
基于 PostgreSQL + pgvector 的混合检索（dense 向量 + sparse 全文检索）。
"""

from __future__ import annotations

from typing import Any

from knowledge.backends import KnowledgeBackend, PgVectorBackend


class KnowledgeBase:

    def __init__(self, long_term_memory: Any):
        if long_term_memory is None:
            raise ValueError("KnowledgeBase 需要 long_term_memory (PostgreSQL + pgvector)，请确保 PG 环境可用")
        self._backend: KnowledgeBackend = PgVectorBackend(long_term_memory)

    @property
    def backend(self) -> KnowledgeBackend:
        return self._backend

    def set_backend(self, backend: KnowledgeBackend) -> None:
        self._backend = backend

    async def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        return await self._backend.search(query, top_k=top_k, score_threshold=score_threshold)

    async def seed(self) -> int:
        return await self._backend.seed()

    async def add_document(
        self, content: str, source: str = "",
        metadata: dict | None = None, category: str = "general",
    ) -> str:
        return await self._backend.add_document(content, source=source, metadata=metadata, category=category)