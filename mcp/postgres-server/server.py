"""
Postgres MCP Server
====================
Safe Postgres querying via MCP. Enforces read-only access,
validates queries before execution, and exposes schema as a resource.

Setup:
    pip install -r requirements.txt
    export DATABASE_URL=postgresql://user:pass@localhost/mydb
    python server.py

Add to Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "postgres": {
          "command": "python",
          "args": ["/path/to/server.py"],
          "env": { "DATABASE_URL": "postgresql://..." }
        }
      }
    }
"""

import asyncio
import os
import re

import asyncpg
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

DATABASE_URL = os.environ.get("DATABASE_URL", "")
server = Server("postgres-mcp-server")

BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


async def get_conn():
    return await asyncpg.connect(DATABASE_URL)


def is_safe(query: str) -> bool:
    return not BLOCKED_KEYWORDS.search(query)


# ── Tools ────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[
        Tool(
            name="query",
            description="Run a read-only SQL query against the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SELECT query to execute"},
                    "limit": {"type": "integer", "default": 50, "maximum": 500},
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="explain",
            description="Get the query plan for a SQL query without executing it",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="list_tables",
            description="List all tables in the database with their row counts",
            inputSchema={"type": "object", "properties": {}},
        ),
    ])


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        conn = await get_conn()

        if name == "query":
            sql = arguments["sql"].strip()
            limit = min(arguments.get("limit", 50), 500)

            if not is_safe(sql):
                return CallToolResult(
                    content=[TextContent(type="text", text="Only SELECT queries are allowed.")],
                    isError=True,
                )

            # Wrap in a limit if not already present
            if "LIMIT" not in sql.upper():
                sql = f"SELECT * FROM ({sql}) AS _q LIMIT {limit}"

            rows = await conn.fetch(sql)
            await conn.close()

            if not rows:
                return CallToolResult(content=[TextContent(type="text", text="No rows returned.")])

            headers = list(rows[0].keys())
            table = " | ".join(headers) + "\n" + "-" * 40
            for row in rows:
                table += "\n" + " | ".join(str(v) for v in row.values())

            return CallToolResult(content=[TextContent(type="text", text=table)])

        elif name == "explain":
            sql = arguments["sql"]
            rows = await conn.fetch(f"EXPLAIN {sql}")
            await conn.close()
            plan = "\n".join(r[0] for r in rows)
            return CallToolResult(content=[TextContent(type="text", text=plan)])

        elif name == "list_tables":
            rows = await conn.fetch("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            await conn.close()
            result = "\n".join(f"{r['table_name']} ({r['size']})" for r in rows)
            return CallToolResult(content=[TextContent(type="text", text=result or "No tables found.")])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {e}")], isError=True)


# ── Resources ────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(resources=[
        Resource(
            uri="postgres://schema",
            name="Database Schema",
            description="Full schema of all public tables",
            mimeType="text/plain",
        )
    ])


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    if uri == "postgres://schema":
        conn = await get_conn()
        rows = await conn.fetch("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        await conn.close()

        schema = {}
        for r in rows:
            schema.setdefault(r["table_name"], []).append(
                f"  {r['column_name']} {r['data_type']} {'NULL' if r['is_nullable'] == 'YES' else 'NOT NULL'}"
            )

        text = "\n\n".join(
            f"TABLE {tbl}:\n" + "\n".join(cols)
            for tbl, cols in schema.items()
        )
        return ReadResourceResult(contents=[TextContent(type="text", text=text)])

    raise ValueError(f"Unknown resource: {uri}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
