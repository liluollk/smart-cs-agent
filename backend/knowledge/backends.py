"""
知识库后端 — 基于 PostgreSQL + pgvector 的混合检索（dense 向量 + sparse 全文检索）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knowledge.seed_data import SEED_DOCUMENTS


class KnowledgeBackend(ABC):
    """知识检索后端抽象基类"""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        """语义检索，返回匹配文档列表"""
        ...

    @abstractmethod
    async def seed(self) -> int:
        """初始化种子数据，返回写入的文档数"""
        ...

    @abstractmethod
    async def add_document(
        self, content: str, source: str = "",
        metadata: dict | None = None, category: str = "general",
    ) -> str:
        """添加单篇文档，返回文档 ID"""
        ...


class PgVectorBackend(KnowledgeBackend):
    """基于 PostgreSQL + pgvector 的混合检索后端
    dense(向量相似度) + sparse(tsvector 全文检索) 加权融合
    """

    def __init__(self, long_term_memory: Any, alpha: float = 0.7):
        self._ltm = long_term_memory
        self._alpha = alpha

    async def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        return await self._ltm.search_hybrid(query, top_k=top_k, score_threshold=score_threshold, alpha=self._alpha)

    async def seed(self) -> int:
        return len(await self._ltm.add_documents_batch([
            {"content": d["content"], "source": d["source"]}
            for d in SEED_DOCUMENTS
        ]))

    async def add_document(
        self, content: str, source: str = "",
        metadata: dict | None = None, category: str = "general",
    ) -> str:
        return await self._ltm.add_document(content, source=source, metadata=metadata, category=category)