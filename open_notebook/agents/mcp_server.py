#!/usr/bin/env python3
"""
MCP Server for Open Notebook RAG Tools.

This server exposes notebook search and retrieval tools via MCP protocol
for use with Claude CLI.
"""

import asyncio
import json
import sys
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from open_notebook.agents.tools import NOTEBOOK_TOOLS, handle_tool_call


# Create the MCP server
app = Server("open-notebook-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available notebook tools."""
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["input_schema"]
        )
        for tool in NOTEBOOK_TOOLS
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """Execute a notebook tool."""
    try:
        result = await handle_tool_call(name, arguments)

        # Extract text content from result
        if result.get("content"):
            text = result["content"][0].get("text", "No result")
        else:
            text = "No result"

        return [TextContent(type="text", text=text)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
