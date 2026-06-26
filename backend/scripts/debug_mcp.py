"""深度调试：直接测试 MCP 工具调用链路"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

async def main():
    from mcps.client import MCPClient
    from mcps.server import MCPServer
    from tools.registry import ToolRegistry

    ltm = None
    mcp_server = MCPServer(name="test")
    tool_registry = ToolRegistry(long_term_memory=ltm)
    tool_registry.register_to_mcp(mcp_server)
    mcp_client = MCPClient(mcp_server)
    await mcp_client.initialize()

    print("=== 直接测试 MCP call_tool ===")
    result = await mcp_client.call_tool("knowledge_search", {"query": "理财产品A"})
    print(f"结果: {result[:500]}")

    print("\n=== MCP Server 调用日志 ===")
    for log in mcp_server.get_call_log(10):
        print(f"  {log}")

asyncio.run(main())