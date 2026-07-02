"""
Hello MCP Server
================
The smallest useful MCP server: one tool, no API keys, no external
services. Built to be read top to bottom in under five minutes.

This is the exact code walked through in the "Build Your First MCP
Server" lesson on AI Learning Hub. It exposes a single tool,
analyze_text, that computes word count, character count, sentence
count, and estimated reading time for a block of text.

Tools:
  - analyze_text(text) — word/character/sentence counts + reading time

Resources:
  - help://usage — how to call the tool

Setup:
    pip install -r requirements.txt
    python server.py

Add to Claude Desktop / Cursor MCP config:
    {
      "mcpServers": {
        "hello-mcp": {
          "command": "python",
          "args": ["/absolute/path/to/server.py"]
        }
      }
    }
"""

import asyncio
import re

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

WORDS_PER_MINUTE = 200

server = Server("hello-mcp-server")


# ── The one tool ─────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[
        Tool(
            name="analyze_text",
            description=(
                "Analyze a block of text and return word count, character count, "
                "sentence count, and estimated reading time in minutes. "
                "Use this whenever the user pastes text and asks how long it is, "
                "how long it takes to read, or wants basic text statistics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze",
                    },
                },
                "required": ["text"],
            },
        ),
    ])


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name != "analyze_text":
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )

    text = arguments.get("text", "")
    if not text.strip():
        return CallToolResult(
            content=[TextContent(type="text", text="No text provided.")],
            isError=True,
        )

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(re.findall(r"[.!?]+", text)) or 1
    reading_time_min = max(1, round(word_count / WORDS_PER_MINUTE))

    summary = (
        f"Words: {word_count}\n"
        f"Characters: {char_count}\n"
        f"Sentences: {sentence_count}\n"
        f"Estimated reading time: {reading_time_min} min "
        f"(at {WORDS_PER_MINUTE} words/min)"
    )
    return CallToolResult(content=[TextContent(type="text", text=summary)])


# ── One help resource ────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(resources=[
        Resource(
            uri="help://usage",
            name="How to use this server",
            description="Quick usage instructions",
            mimeType="text/plain",
        )
    ])


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    if uri == "help://usage":
        return ReadResourceResult(
            contents=[TextContent(
                type="text",
                text="Call analyze_text with any block of text to get word "
                     "count, character count, sentence count, and estimated "
                     "reading time.",
            )]
        )
    raise ValueError(f"Unknown resource: {uri}")


# ── Entry point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
