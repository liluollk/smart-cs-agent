"""
长期记忆 — 基于 PostgreSQL + pgvector 的向量存储
存储用户画像、历史工单、知识库文档等需要持久化的信息。
支持语义相似度检索，用于 RAG 知识检索 Agent。

改进：
- asyncpg 异步连接池，不阻塞事件循环
- Embedding 结果 LRU 缓存，减少 API 调用
- 完善的日志和错误处理
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

try:
    import asyncpg
    from pgvector.asyncpg import register_vector as _pgv_register
except ImportError:
    asyncpg = None
    _pgv_register = None

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class _LRUCache:
    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> list[float] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value


class LongTermMemory:
    """
    长期记忆：基于 PostgreSQL + pgvector 的向量存储。

    特点：
    - 向量化存储，支持语义相似度检索
    - 持久化到 PostgreSQL，跨会话保持
    - 支持增量更新和批量导入
    - 使用百炼 text-embedding-v4 模型生成向量（1024 维）
    - 父子块机制：子块用于检索（精准），父块用于生成（完整上下文）
    - HyDE 查询优化：短查询先生成假想答案再检索
    - Cross-Encoder 重排序：粗筛 + 精排两阶段检索
    - asyncpg 异步连接池，不阻塞事件循环
    - Embedding 结果 LRU 缓存，减少 API 调用
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "smartcs",
        user: str = "postgres",
        password: str = "",
        dashscope_api_key: str = "",
        dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_parent_child: bool = True,
        enable_hyde: bool = True,
        enable_rerank: bool = True,
        hyde_llm: Any = None,
        rerank_client: Any = None,
        embedding_cache_size: int = 256,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._pool: asyncpg.Pool | None = None

        self._embedding_client = AsyncOpenAI(
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
        )
        self._embedding_cache = _LRUCache(max_size=embedding_cache_size)

        self._enable_parent_child = enable_parent_child
        self._enable_hyde = enable_hyde
        self._hyde_llm = hyde_llm
        self._enable_rerank = enable_rerank
        self._rerank_client = rerank_client

        if asyncpg is None or _pgv_register is None:
            raise ImportError("请安装 pgvector + asyncpg: pip install pgvector asyncpg")

    async def connect(self):
        if self._pool is None:
            async def _init_conn(conn):
                await _pgv_register(conn)

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                init=_init_conn,
                min_size=2,
                max_size=10,
            )
            logger.info("PostgreSQL 连接池已建立 (%s:%d/%s)", self.host, self.port, self.database)

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL 连接池已关闭")

    async def _get_embedding(self, text: str) -> list[float]:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = await self._embedding_client.embeddings.create(
                input=text, model="text-embedding-v4"
            )
            embedding = resp.data[0].embedding
            self._embedding_cache.put(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.error("Embedding 生成失败: %s", e)
            raise

    async def add_document(
        self, content: str, source: str = "", metadata: dict | None = None, category: str = "general"
    ) -> str:
        await self.connect()

        if self._enable_parent_child:
            return await self._add_document_with_parent_child(content, source, metadata, category)

        embedding = await self._get_embedding(content)
        doc_id = self._generate_doc_id(content)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO knowledge_base (id, content, source, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    doc_id, content, source, metadata_json, embedding,
                )
                title = source.replace(".md", "").replace(".txt", "").replace("_", " ").title() if source else f"文档_{doc_id}"
                await conn.execute(
                    """
                    INSERT INTO kb_documents (doc_id, title, category, content, source, embedding, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    """,
                    doc_id, title, category, content, source, embedding, "system",
                )

        logger.info("文档已添加: %s (source=%s)", doc_id, source)
        return doc_id

    async def _add_document_with_parent_child(
        self, content: str, source: str = "", metadata: dict | None = None, category: str = "general"
    ) -> str:
        parent_id = self._generate_doc_id(content)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        parent_embedding = await self._get_embedding(content)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO knowledge_base (id, content, source, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    parent_id, content, source, metadata_json, parent_embedding,
                )

                title = source.replace(".md", "").replace(".txt", "").replace("_", " ").title() if source else f"文档_{parent_id}"
                await conn.execute(
                    """
                    INSERT INTO kb_documents (doc_id, title, category, content, source, embedding, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    """,
                    parent_id, title, category, content, source, parent_embedding, "system",
                )

                child_chunks = self._chunk_text(content, chunk_size=256, overlap=64)
                for i, chunk in enumerate(child_chunks):
                    child_id = f"{parent_id}_child_{i}"
                    child_embedding = await self._get_embedding(chunk)
                    child_meta = json.dumps({"parent_id": parent_id, "chunk_index": i}, ensure_ascii=False)
                    await conn.execute(
                        """
                        INSERT INTO knowledge_base (id, content, source, metadata, embedding)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            source = EXCLUDED.source,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        child_id, chunk, source, child_meta, child_embedding,
                    )

        logger.info("父子块文档已添加: %s (%d 子块, source=%s)", parent_id, len(child_chunks), source)
        return parent_id

    async def add_documents_batch(self, documents: list[dict]) -> list[str]:
        doc_ids = []
        for doc in documents:
            doc_id = await self.add_document(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                metadata=doc.get("metadata"),
                category=doc.get("category", "general"),
            )
            doc_ids.append(doc_id)
        logger.info("批量添加文档完成: %d 篇", len(doc_ids))
        return doc_ids

    async def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        return await self.search_hybrid(query, top_k=top_k, score_threshold=score_threshold, alpha=1.0)

    async def search_hybrid(self, query: str, top_k: int = 5, score_threshold: float = 0.0, alpha: float = 0.7) -> list[dict]:
        await self.connect()

        effective_query = await self._apply_hyde(query) if self._enable_hyde else query
        query_embedding = await self._get_embedding(effective_query)
        coarse_top_k = top_k * 10 if self._enable_rerank else top_k

        async with self._pool.acquire() as conn:
            if alpha >= 1.0:
                rows = await conn.fetch(
                    """
                    SELECT id, content, source, metadata,
                           1 - (embedding <=> $1::vector) AS score
                    FROM knowledge_base
                    WHERE 1 - (embedding <=> $1::vector) > 0
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding, coarse_top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, content, source, metadata,
                           (1 - (embedding <=> $1::vector)) * $2
                           + COALESCE(ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', $3)), 0) * $4
                           AS score
                    FROM knowledge_base
                    WHERE 1 - (embedding <=> $1::vector) > 0
                    ORDER BY score DESC
                    LIMIT $5
                    """,
                    query_embedding, alpha, query, 1 - alpha, coarse_top_k,
                )

        results = []
        for row in rows:
            score = float(row["score"])
            if score < score_threshold:
                continue
            results.append({
                "id": row["id"],
                "content": row["content"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "score": round(score, 3),
            })

        if self._enable_parent_child:
            results = await self._map_children_to_parents(results)

        if self._enable_rerank and self._rerank_client:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]

        logger.debug("混合检索完成: query=%r, top_k=%d, 返回 %d 条", query, top_k, len(results))
        return results

    async def _apply_hyde(self, query: str) -> str:
        if not self._hyde_llm or len(query) > 20:
            return query

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            hyde_prompt = (
                "你是一个检索助手。根据用户问题，生成一个可能的答案片段。\n"
                "这个答案片段将用于检索相关文档。只输出简短的事实性描述，不要解释。\n"
                f"用户问题: {query}\n可能的答案片段:"
            )

            response = await self._hyde_llm.ainvoke([
                SystemMessage(content="你只输出简短的事实性描述，不超过 50 字。"),
                HumanMessage(content=hyde_prompt),
            ])
            hypothetical = response.content.strip()
            if hypothetical and len(hypothetical) < 200:
                return f"{query} {hypothetical}"
        except Exception as e:
            logger.warning("HyDE 生成失败: %s", e)

        return query

    async def _map_children_to_parents(self, results: list[dict]) -> list[dict]:
        parent_map: dict[str, dict] = {}

        async with self._pool.acquire() as conn:
            for doc in results:
                meta = doc.get("metadata", {})
                parent_id = meta.get("parent_id")

                if parent_id:
                    if parent_id not in parent_map:
                        row = await conn.fetchrow(
                            "SELECT id, content, source, metadata FROM knowledge_base WHERE id = $1",
                            parent_id,
                        )
                        if row:
                            parent_map[parent_id] = {
                                "id": row["id"],
                                "content": row["content"],
                                "source": row["source"],
                                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                                "score": doc["score"],
                            }
                else:
                    if doc["id"] not in parent_map:
                        parent_map[doc["id"]] = doc

        return sorted(parent_map.values(), key=lambda x: x["score"], reverse=True)

    def _rerank_results(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if not results or not self._rerank_client:
            return results[:top_k]

        try:
            pairs = [[query, doc["content"]] for doc in results]
            scores = self._rerank_client.compute_score(pairs)

            if isinstance(scores, float):
                scores = [scores]

            for i, doc in enumerate(results):
                doc["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

            results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.warning("重排序失败: %s", e)
            return results[:top_k]

    @staticmethod
    def _generate_doc_id(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def load_knowledge_base(self, kb_dir: str) -> int:
        from pathlib import Path

        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return 0

        count = 0
        for file_path in kb_path.glob("**/*.txt"):
            content = file_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(content)
            for chunk in chunks:
                await self.add_document(
                    content=chunk,
                    source=str(file_path.name),
                    metadata={"file": str(file_path)},
                )
                count += 1

        logger.info("知识库加载完成: %d 块 (dir=%s)", count, kb_dir)
        return count

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
                else:
                    sentences = para.replace("。", "。\n").replace(".", ".\n").split("\n")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:chunk_size]]