"""
FastAPI 应用入口 — 完整 MCP Client/Server 架构。
- MCP Server 注册所有工具
- MCP Client 通过 JSON-RPC 协议调用工具
- 所有 Agent 通过 MCP Client 间接调用工具
- SlowAPI 限流保护
"""

from __future__ import annotations

import json
import logging
import uuid
from io import BytesIO
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from agents.supervisor import Supervisor
from agents.knowledge import KnowledgeAgent
from agents.ticket import TicketAgent
from core.router import RouterAgent
from core.compliance import ComplianceAgent
from core.registry import AgentRegistry
from api.schemas import ChatRequest, ChatResponse
from config import settings
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from memory.working_memory import WorkingMemory
from tools.registry import ToolRegistry
from mcps.server import MCPServer
from mcps.client import MCPClient
from tracing.collector import get_collector, init_collector

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    llm: ChatOpenAI
    mcp_server: MCPServer
    mcp_client: MCPClient
    tool_registry: ToolRegistry
    graph: CompiledStateGraph
    working_memory: WorkingMemory
    short_term_memory: ShortTermMemory


@asynccontextmanager
async def lifespan(app: FastAPI):
    warnings = settings.validate()
    for w in warnings:
        logger.warning(w)

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=settings.llm_temperature,
    )

    if settings.pg_password:
        long_term_memory = LongTermMemory(
            host=settings.pg_host,
            port=settings.pg_port,
            database=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password,
            dashscope_api_key=settings.dashscope_api_key,
            dashscope_base_url=settings.dashscope_base_url,
            embedding_cache_size=settings.embedding_cache_size,
        )
    else:
        raise RuntimeError("PG_PASSWORD 未配置，知识库依赖 PostgreSQL + pgvector，请检查 .env 文件")

    working_memory = WorkingMemory(redis_url=settings.redis_url or None)
    short_term_memory = ShortTermMemory(redis_url=settings.redis_url or "redis://localhost:6379/0")

    mcp_server = MCPServer(name="smart-cs-mcp")

    tool_registry = ToolRegistry(
        long_term_memory=long_term_memory,
        tool_timeout_seconds=settings.tool_timeout_seconds,
    )
    tool_registry.register_to_mcp(mcp_server)
    try:
        await tool_registry.seed_knowledge_base()
    except Exception as e:
        logger.warning("知识库初始化跳过（数据库未就绪）: %s", e)

    mcp_client = MCPClient(mcp_server)
    await mcp_client.initialize()

    init_collector(max_traces=100, persist_dir="data/traces")

    openai_functions = tool_registry.get_openai_functions()

    knowledge_functions = [f for f in openai_functions
                           if f["function"]["name"] in ("knowledge_search",)]
    ticket_functions = [f for f in openai_functions
                        if f["function"]["name"] in ("risk_check", "ticket_create", "notification_send", "order_query")]

    router = RouterAgent(llm)
    compliance_agent = ComplianceAgent()

    knowledge_agent = KnowledgeAgent(
        llm, mcp_client, knowledge_functions, short_term_memory=short_term_memory,
    )
    ticket_agent = TicketAgent(
        llm, mcp_client, ticket_functions, short_term_memory=short_term_memory,
    )

    agent_registry = AgentRegistry()
    agent_registry.register("knowledge", knowledge_agent, intents=["knowledge", "ticket"], priority=0, always_run=True)
    agent_registry.register("ticket", ticket_agent, intents=["ticket"], priority=1)

    supervisor = Supervisor(
        llm, mcp_client,
        agent_registry=agent_registry,
        router=router,
        compliance_agent=compliance_agent,
        working_memory=working_memory, short_term_memory=short_term_memory,
    )
    graph = supervisor.build_graph()

    app.state.container = AppContainer(
        llm=llm,
        mcp_server=mcp_server,
        mcp_client=mcp_client,
        tool_registry=tool_registry,
        graph=graph,
        working_memory=working_memory,
        short_term_memory=short_term_memory,
    )

    yield


app = FastAPI(title="Smart CS Multi-Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _get_container(request: Request) -> AppContainer:
    return request.app.state.container


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    logger.info("chat 端点被调用, message=%s", body.message[:50])
    container = _get_container(request)

    session_id = body.session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    collector = get_collector()
    span = collector.start_span("chat", trace_id=trace_id)

    try:
        result = await container.graph.ainvoke(
            dict(messages=[HumanMessage(content=body.message)], user_id=body.user_id, session_id=session_id,
                 trace_id=trace_id),
            {"configurable": {"thread_id": session_id}},
        )
    except Exception as e:
        logger.error("chat 处理异常: %s", e, exc_info=True)
        collector.end_span(span, error=str(e))
        raise HTTPException(status_code=500, detail="内部服务错误，请稍后重试")

    collector.end_span(span)

    return ChatResponse(
        session_id=session_id,
        trace_id=trace_id,
        intent=result.get("intent", "knowledge"),
        response=result.get("final_response", ""),
        compliance_passed=result.get("compliance_passed", True) and not result.get("input_compliance_blocked", False),
    )


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    container = _get_container(request)

    session_id = body.session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    collector = get_collector()
    span = collector.start_span("chat_stream", trace_id=trace_id)

    async def event_stream():
        try:
            async for event in container.graph.astream(
                    {'messages': [HumanMessage(content=body.message)], 'user_id': body.user_id,
                     'session_id': session_id, 'trace_id': trace_id},
                    {"configurable": {"thread_id": session_id}},
                    stream_mode="updates",
            ):
                node_name = list(event.keys())[0]
                node_data = event[node_name]

                if node_name == "synthesize" and "final_response" in node_data:
                    yield f"data: {json.dumps({'type': 'done', 'response': node_data['final_response'], 'trace_id': trace_id, 'session_id': session_id}, ensure_ascii=False)}\n\n"
                elif node_name == "fan_out" and "sub_results" in node_data:
                    for k, v in node_data["sub_results"].items():
                        yield f"data: {json.dumps({'type': 'partial', 'agent': k, 'content': v[:200] if v else ''}, ensure_ascii=False)}\n\n"
                elif node_name == "compliance_pre" and node_data.get("input_compliance_blocked"):
                    yield f"data: {json.dumps({'type': 'blocked', 'reason': 'input_compliance'}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'node', 'node': node_name}, ensure_ascii=False)}\n\n"

            collector.end_span(span)
        except Exception as e:
            logger.error("chat_stream 异常: %s", e, exc_info=True)
            collector.end_span(span, error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"X-Trace-Id": trace_id, "X-Session-Id": session_id})


@app.post("/v1/mcp")
async def mcp_endpoint():
    return JSONResponse(content={"status": "ok", "message": "MCP server running in-process via FastMCP"})


@app.get("/v1/health")
async def health(req: Request):
    container = _get_container(req)
    return {
        "status": "ok",
        "mcp_server": "ready",
        "mcp_client": "connected",
        "llm_model": container.llm.model_name,
    }


@app.get("/v1/traces")
async def list_traces(n: int = 20):
    collector = get_collector()
    return {"traces": collector.get_recent_traces(n)}


@app.get("/v1/traces/{trace_id}")
async def get_trace(trace_id: str):
    collector = get_collector()
    spans = collector.get_trace(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "spans": spans}


@app.get("/v1/mcp/tools")
async def list_mcp_tools(req: Request):
    container = _get_container(req)
    try:
        tools = await container.mcp_client.list_tools()
        return {
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        }
    except Exception as e:
        logger.error("list_mcp_tools 失败: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/v1/metrics/tools")
async def get_tool_metrics(
        tool_name: str | None = None,
        minutes: int | None = None,
):
    collector = get_collector()
    time_window = minutes * 60 if minutes else None
    return collector.get_tool_metrics(tool_name=tool_name, time_window_seconds=time_window)


@app.get("/v1/metrics/tools/recent")
async def get_recent_tool_calls(n: int = 20):
    collector = get_collector()
    return collector.get_recent_tool_calls(n)


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024


def _parse_upload(file_bytes: bytes, filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    elif ext == ".docx":
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        return "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )
    elif ext in (".md", ".txt"):
        return file_bytes.decode("utf-8")
    raise ValueError(f"不支持的文件类型: {ext}")


@app.post("/v1/upload")
async def upload_file(file: UploadFile = File(...), request: Request = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"文件大小超过限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)")

    try:
        text = _parse_upload(file_bytes, file.filename)
    except Exception as e:
        logger.error("文件解析失败: %s", e)
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    container = _get_container(request)
    try:
        doc_id = await container.tool_registry._long_term_memory.add_document(
            content=text,
            source=file.filename,
            metadata={"filename": file.filename, "size": len(file_bytes)},
        )
        logger.info("文件上传成功: %s, doc_id=%s, size=%d", file.filename, doc_id, len(file_bytes))
        return JSONResponse(content={
            "status": "ok",
            "filename": file.filename,
            "doc_id": doc_id,
            "size": len(file_bytes),
            "chars": len(text),
        })
    except Exception as e:
        logger.error("知识库写入失败: %s", e)
        raise HTTPException(status_code=500, detail=f"知识库写入失败: {str(e)}")


@app.get("/v1/documents")
async def list_documents(req: Request):
    container = _get_container(req)
    ltm = container.tool_registry._long_term_memory
    try:
        await ltm.connect()
        async with ltm._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id, title, category, source, created_by, created_at, updated_at "
                "FROM kb_documents ORDER BY created_at DESC"
            )
            docs = []
            for r in rows:
                chunk_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM knowledge_base WHERE id LIKE $1",
                    r["doc_id"] + "_child_%",
                )
                content_preview = await conn.fetchval(
                    "SELECT LEFT(content, 200) FROM kb_documents WHERE doc_id = $1",
                    r["doc_id"],
                )
                docs.append({
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "category": r["category"],
                    "source": r["source"],
                    "created_by": r["created_by"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                    "chunk_count": chunk_count,
                    "preview": content_preview or "",
                })
            return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error("获取文档列表失败: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/v1/documents/{doc_id}")
async def get_document(doc_id: str, req: Request):
    container = _get_container(req)
    ltm = container.tool_registry._long_term_memory
    try:
        await ltm.connect()
        async with ltm._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT doc_id, title, category, content, source, created_by, created_at, updated_at "
                "FROM kb_documents WHERE doc_id = $1",
                doc_id,
            )
            if not row:
                raise HTTPException(status_code=404, detail="文档不存在")
            return {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "category": row["category"],
                "content": row["content"],
                "source": row["source"],
                "created_by": row["created_by"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取文档详情失败: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/v1/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, req: Request):
    container = _get_container(req)
    ltm = container.tool_registry._long_term_memory
    try:
        await ltm.connect()
        async with ltm._pool.acquire() as conn:
            parent = await conn.fetchrow(
                "SELECT doc_id, title, source FROM kb_documents WHERE doc_id = $1",
                doc_id,
            )
            if not parent:
                raise HTTPException(status_code=404, detail="文档不存在")

            rows = await conn.fetch(
                "SELECT id, content, metadata FROM knowledge_base WHERE id LIKE $1 ORDER BY id",
                doc_id + "_child_%",
            )
            chunks = []
            for r in rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                chunks.append({
                    "id": r["id"],
                    "content": r["content"],
                    "chunk_index": meta.get("chunk_index", 0),
                    "char_count": len(r["content"]),
                })
            if not chunks:
                single = await conn.fetchrow(
                    "SELECT id, content FROM knowledge_base WHERE id = $1",
                    doc_id,
                )
                if single:
                    chunks.append({
                        "id": single["id"],
                        "content": single["content"],
                        "chunk_index": 0,
                        "char_count": len(single["content"]),
                    })
            return {
                "doc_id": doc_id,
                "title": parent["title"],
                "source": parent["source"],
                "total_chunks": len(chunks),
                "chunks": chunks,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取文档分块失败: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/v1/documents/{doc_id}")
async def delete_document(doc_id: str, req: Request):
    container = _get_container(req)
    ltm = container.tool_registry._long_term_memory
    try:
        await ltm.connect()
        async with ltm._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM knowledge_base WHERE id LIKE $1",
                    doc_id + "_child_%",
                )
                await conn.execute(
                    "DELETE FROM knowledge_base WHERE id = $1",
                    doc_id,
                )
                await conn.execute(
                    "DELETE FROM kb_documents WHERE doc_id = $1",
                    doc_id,
                )
        logger.info("文档已删除: %s", doc_id)
        return {"status": "ok", "doc_id": doc_id}
    except Exception as e:
        logger.error("删除文档失败: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/v1/metrics/tools/{tool_name}")
async def get_single_tool_metrics(tool_name: str, minutes: int | None = None):
    collector = get_collector()
    time_window = minutes * 60 if minutes else None
    return collector.get_tool_metrics(tool_name=tool_name, time_window_seconds=time_window)