"""
MCP 客户端 — 基于 FastMCP 官方 SDK，通过 Client 与 MCP Server 通信。
支持进程内直连（in-process）和 HTTP 远程两种传输模式。

所有工具调用均通过 MCP 协议，实现真正的 MCP Client/Server 架构。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastmcp import Client as FastMCPClient

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """MCP 工具元信息"""
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    """
    MCP 客户端 — 负责与 MCP Server 通信，提供工具发现和调用能力。

    用法:
        server = MCPServer("my-server")
        client = MCPClient(server)
        await client.initialize()
        tools = await client.list_tools()
        result = await client.call_tool("knowledge_search", {"query": "收益率"})
    """

    def __init__(self, server_or_url):
        if isinstance(server_or_url, str):
            self._client = FastMCPClient(server_or_url)
        else:
            self._client = FastMCPClient(server_or_url.inner)
        self._ctx = None

    async def initialize(self) -> dict:
        try:
            self._ctx = await self._client.__aenter__()
            logger.info("MCP 客户端初始化成功")
            return {}
        except Exception as e:
            logger.error("MCP 客户端初始化失败: %s", e)
            raise

    async def ping(self) -> bool:
        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.warning("MCP ping 失败: %s", e)
            return False

    async def list_tools(self) -> list[MCPToolInfo]:
        try:
            tools = await self._client.list_tools()
            return [
                MCPToolInfo(
                    name=t.name,
                    description=getattr(t, "description", ""),
                    input_schema=getattr(t, "inputSchema", {}),
                )
                for t in tools
            ]
        except Exception as e:
            logger.error("MCP list_tools 失败: %s", e)
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            result = await self._client.call_tool(name, arguments)
            if hasattr(result, "content") and result.content:
                texts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                return "\n".join(texts) if texts else str(result)
            return str(result)
        except Exception as e:
            logger.error("MCP call_tool(%s) 失败: %s", name, e)
            return f"[工具调用错误] {name}: {e}"

    async def close(self):
        if self._ctx is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("MCP 客户端关闭异常: %s", e)
            finally:
                self._ctx = None