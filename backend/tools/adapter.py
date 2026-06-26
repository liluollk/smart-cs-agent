"""
工具适配器 — 统一通过 MCP 协议调用工具。
handler 注册到 MCP Server，Agent 通过 MCP Client 直接调用。

设计思想：
- 每个工具自带元数据（风险等级、超时、是否需要人工审核）
- 永远走 MCP：Agent → MCP Client → JSON-RPC → MCP Server → handler
- 支持同步/异步 handler 自动适配
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolMeta:
    """工具元数据"""
    name: str
    description: str
    risk_level: str = "low"
    timeout_seconds: float = 10.0
    requires_approval: bool = False
    retry_count: int = 1
    category: str = "general"


class ToolAdapter:
    """
    工具适配器：handler 给 MCP Server 执行，Agent 通过 MCP Client 直接调用。

    用法:
        def my_handler(query: str) -> str:
            return json.dumps({"result": f"search: {query}"})

        adapter = ToolAdapter(
            meta=ToolMeta(name="my_search", description="搜索"),
            handler=my_handler,
        )
        registry.register(adapter)
    """

    def __init__(self, meta: ToolMeta, handler: Callable):
        self.meta = meta
        self._handler = handler

    def to_openai_function(self) -> dict:
        """生成 OpenAI function calling 格式，供 LLM 识别可调用工具"""
        schema = self._infer_schema()
        properties = {}
        required = []

        for name, info in schema.items():
            json_type = self._schema_type_to_json(info["type"])
            prop: dict[str, Any] = {"type": json_type, "description": f"{name} 参数"}
            if info["default"] is not inspect.Parameter.empty:
                prop["default"] = info["default"]
            else:
                required.append(name)
            properties[name] = prop

        return {
            "type": "function",
            "function": {
                "name": self.meta.name,
                "description": self.meta.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_mcp_spec(self) -> dict:
        """生成 MCP Server 注册用的工具规范"""
        return {
            "name": self.meta.name,
            "description": self.meta.description,
            "input_schema": {name: info["type"] for name, info in self._infer_schema().items()},
            "handler": self._handler,
        }

    def _infer_schema(self) -> dict[str, dict[str, Any]]:
        sig = inspect.signature(self._handler)
        type_map = {int: "int", float: "float", bool: "bool", str: "str"}
        schema = {}
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            anno = param.annotation
            type_str = type_map.get(anno, "str") if anno is not inspect.Parameter.empty else "str"
            schema[name] = {"type": type_str, "default": param.default}
        return schema

    @staticmethod
    def _schema_type_to_json(type_str: str) -> str:
        type_map = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}
        return type_map.get(type_str, "string")