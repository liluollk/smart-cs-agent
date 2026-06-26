from __future__ import annotations

import logging
from typing import Any, Callable

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server — 基于 FastMCP 官方 SDK，提供工具注册和调用能力"""

    def __init__(self, name: str = "smart-cs-mcp"):
        self._mcp = FastMCP(name)
        self._call_log: list[dict] = []
        self._registered_tools: list[dict[str, Any]] = []

    def register_tool(self, name: str, description: str, handler: Callable):
        self._mcp.tool(name=name, description=description)(handler)
        self._registered_tools.append({
            "name": name,
            "description": description,
        })
        logger.info("MCP 工具注册: %s", name)

    @property
    def inner(self) -> FastMCP:
        return self._mcp

    def get_call_log(self, last_n: int = 100) -> list[dict]:
        return self._call_log[-last_n:]

    def list_registered_tools(self) -> list[dict[str, Any]]:
        return list(self._registered_tools)