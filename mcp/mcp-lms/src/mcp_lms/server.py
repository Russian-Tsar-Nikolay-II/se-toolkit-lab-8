"""Stdio MCP server exposing LMS backend operations as typed tools."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field

from mcp_lms.client import LMSClient


class _NoArgs(BaseModel):
    """Empty input model for tools that only need server-side configuration."""


class _LabQuery(BaseModel):
    lab: str = Field(description="Lab identifier, e.g. 'lab-04'.")


class _TopLearnersQuery(_LabQuery):
    limit: int = Field(
        default=5, ge=1, description="Max learners to return (default 5)."
    )


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    api_key: str


ToolPayload = BaseModel | Sequence[BaseModel]
ToolHandler = Callable[[LMSClient, BaseModel], Awaitable[ToolPayload]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    handler: ToolHandler

    def as_tool(self) -> Tool:
        schema = self.model.model_json_schema()
        schema.pop("$defs", None)
        schema.pop("title", None)
        return Tool(name=self.name, description=self.description, inputSchema=schema)


def _resolve_api_key() -> str:
    for name in ("NANOBOT_LMS_API_KEY", "LMS_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "LMS API key not configured. Set NANOBOT_LMS_API_KEY or LMS_API_KEY."
    )


def _resolve_base_url(base_url: str | None = None) -> str:
    value = (base_url or os.environ.get("NANOBOT_LMS_BACKEND_URL", "")).strip()
    if not value:
        raise RuntimeError(
            "LMS backend URL not configured. Pass it as: python -m mcp_lms <base_url>"
        )
    return value


def _resolve_settings(base_url: str | None = None) -> Settings:
    return Settings(base_url=_resolve_base_url(base_url), api_key=_resolve_api_key())


def _text(data: BaseModel | Sequence[BaseModel]) -> list[TextContent]:
    if isinstance(data, BaseModel):
        payload = data.model_dump()
    else:
        payload = [item.model_dump() for item in data]
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def _health(client: LMSClient, _args: BaseModel) -> ToolPayload:
    return await client.health_check()


async def _labs(client: LMSClient, _args: BaseModel) -> ToolPayload:
    return await client.get_labs()


async def _learners(client: LMSClient, _args: BaseModel) -> ToolPayload:
    return await client.get_learners()


async def _pass_rates(client: LMSClient, args: BaseModel) -> ToolPayload:
    return await client.get_pass_rates(_require_lab_query(args).lab)


async def _timeline(client: LMSClient, args: BaseModel) -> ToolPayload:
    return await client.get_timeline(_require_lab_query(args).lab)


async def _groups(client: LMSClient, args: BaseModel) -> ToolPayload:
    return await client.get_groups(_require_lab_query(args).lab)


async def _top_learners(client: LMSClient, args: BaseModel) -> ToolPayload:
    query = _require_top_learners_query(args)
    return await client.get_top_learners(query.lab, limit=query.limit)


async def _completion_rate(client: LMSClient, args: BaseModel) -> ToolPayload:
    return await client.get_completion_rate(_require_lab_query(args).lab)


async def _sync_pipeline(client: LMSClient, _args: BaseModel) -> ToolPayload:
    return await client.sync_pipeline()


def _require_lab_query(args: BaseModel) -> _LabQuery:
    if not isinstance(args, _LabQuery):
        raise TypeError(f"Expected {_LabQuery.__name__}, got {type(args).__name__}")
    return args


def _require_top_learners_query(args: BaseModel) -> _TopLearnersQuery:
    if not isinstance(args, _TopLearnersQuery):
        raise TypeError(
            f"Expected {_TopLearnersQuery.__name__}, got {type(args).__name__}"
        )
    return args


_TOOL_SPECS = (
    ToolSpec(
        "lms_health",
        "Check if the LMS backend is healthy and report the item count.",
        _NoArgs,
        _health,
    ),
    ToolSpec("lms_labs", "List all labs available in the LMS.", _NoArgs, _labs),
    ToolSpec(
        "lms_learners",
        "List all learners registered in the LMS.",
        _NoArgs,
        _learners,
    ),
    ToolSpec(
        "lms_pass_rates",
        "Get pass rates (avg score and attempt count per task) for a lab.",
        _LabQuery,
        _pass_rates,
    ),
    ToolSpec(
        "lms_timeline",
        "Get submission timeline (date + submission count) for a lab.",
        _LabQuery,
        _timeline,
    ),
    ToolSpec(
        "lms_groups",
        "Get group performance (avg score + student count per group) for a lab.",
        _LabQuery,
        _groups,
    ),
    ToolSpec(
        "lms_top_learners",
        "Get top learners by average score for a lab.",
        _TopLearnersQuery,
        _top_learners,
    ),
    ToolSpec(
        "lms_completion_rate",
        "Get completion rate (passed / total) for a lab.",
        _LabQuery,
        _completion_rate,
    ),
    ToolSpec(
        "lms_sync_pipeline",
        "Trigger the LMS sync pipeline. May take a moment.",
        _NoArgs,
        _sync_pipeline,
    ),
)
_TOOLS = {spec.name: spec for spec in _TOOL_SPECS}


def create_server(client: LMSClient) -> Server:
    server = Server("lms")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [spec.as_tool() for spec in _TOOL_SPECS]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        spec = _TOOLS.get(name)
        if spec is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            args = spec.model.model_validate(arguments or {})
            return _text(await spec.handler(client, args))
        except Exception as exc:
            return [
                TextContent(type="text", text=f"Error: {type(exc).__name__}: {exc}")
            ]

    _ = list_tools, call_tool
    return server


async def main(base_url: str | None = None) -> None:
    settings = _resolve_settings(base_url)
    async with LMSClient(settings.base_url, settings.api_key) as client:
        server = create_server(client)
        async with stdio_server() as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
