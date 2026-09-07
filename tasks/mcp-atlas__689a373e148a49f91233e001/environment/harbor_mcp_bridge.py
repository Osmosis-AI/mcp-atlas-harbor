#!/usr/bin/env python3
"""Expose MCP-Atlas's REST API as an allowlisted MCP server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


DEFAULT_ALLOWLIST_PATH = Path("/harbor/enabled_tools.json")
DEFAULT_BASE_URL = "http://mcp-atlas-runtime:1984"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18765
DEFAULT_TIMEOUT_SEC = 180.0
DEFAULT_STARTUP_TIMEOUT_SEC = 240.0
DEFAULT_STARTUP_RETRY_INTERVAL_SEC = 1.0


class BridgeConfigurationError(ValueError):
    """The gateway configuration is invalid."""


class AtlasBackendError(RuntimeError):
    """The Atlas REST backend failed or returned invalid data."""


class ToolNotAllowedError(ValueError):
    """A caller requested a tool outside the task allowlist."""


def load_allowlist(path: Path = DEFAULT_ALLOWLIST_PATH) -> tuple[str, ...]:
    """Load the exact, ordered tool allowlist generated for this task."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeConfigurationError("Tool allowlist must be valid JSON.") from exc
    if not isinstance(value, list) or not value:
        raise BridgeConfigurationError("Tool allowlist must be a non-empty array.")
    if any(
        not isinstance(item, str) or not item or item.strip() != item for item in value
    ):
        raise BridgeConfigurationError("Tool names must be non-empty, trimmed strings.")
    if len(value) != len(set(value)):
        raise BridgeConfigurationError("Tool allowlist contains duplicate names.")
    return tuple(value)


class AtlasRESTClient:
    """Async client for the three Atlas REST endpoints used by the bridge."""

    _ENDPOINTS = frozenset({"/call-tool", "/health", "/list-tools"})

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        try:
            parsed = urlsplit(base_url)
            _ = parsed.port
        except ValueError as exc:
            raise BridgeConfigurationError("MCP_ATLAS_BASE_URL is invalid.") from exc
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise BridgeConfigurationError("MCP_ATLAS_BASE_URL must be an HTTP origin.")
        if timeout_sec <= 0:
            raise BridgeConfigurationError("MCP_ATLAS_TIMEOUT_SEC must be positive.")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.transport = transport

    async def request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if endpoint not in self._ENDPOINTS or method not in {"GET", "POST"}:
            raise BridgeConfigurationError("Unsupported Atlas REST request.")
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_sec,
                transport=self.transport,
            ) as client:
                response = await client.request(
                    method,
                    endpoint,
                    json=dict(payload) if payload is not None else None,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AtlasBackendError("Atlas backend request failed.") from exc


class AtlasClient(Protocol):
    async def request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any: ...


_CONTENT_ADAPTER = TypeAdapter(types.ContentBlock)


def _tool_from_json(value: Any) -> types.Tool:
    if not isinstance(value, Mapping):
        raise AtlasBackendError("Atlas returned an invalid tool.")
    data = dict(value)
    # Atlas returns only content blocks from /call-tool.
    data.pop("outputSchema", None)
    try:
        return types.Tool.model_validate(data)
    except ValidationError as exc:
        raise AtlasBackendError("Atlas returned an invalid tool.") from exc


def _content_from_json(value: Any) -> types.ContentBlock:
    try:
        return _CONTENT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise AtlasBackendError("Atlas returned invalid tool content.") from exc


class MCPAtlasBridge:
    """Expose exactly the tools enabled by the current task."""

    def __init__(self, allowed_tools: Sequence[str], client: AtlasClient) -> None:
        self.allowed_tools = tuple(allowed_tools)
        self._allowed_set = frozenset(self.allowed_tools)
        if not self.allowed_tools or len(self.allowed_tools) != len(self._allowed_set):
            raise BridgeConfigurationError(
                "Allowed tools must be non-empty and unique."
            )
        self.client = client
        self._advertised_tools: tuple[types.Tool, ...] = ()

    async def refresh_tools(self) -> tuple[types.Tool, ...]:
        response = await self.client.request_json("POST", "/list-tools")
        if not isinstance(response, list):
            raise AtlasBackendError("Atlas list-tools response must be an array.")
        available: dict[str, types.Tool] = {}
        for value in response:
            tool = _tool_from_json(value)
            if tool.name in available:
                raise AtlasBackendError("Atlas returned duplicate tool names.")
            available[tool.name] = tool
        if self._allowed_set - available.keys():
            raise AtlasBackendError("Atlas backend is missing an allowlisted tool.")
        self._advertised_tools = tuple(available[name] for name in self.allowed_tools)
        return self._advertised_tools

    async def list_tools(self) -> list[types.Tool]:
        if not self._advertised_tools:
            await self.refresh_tools()
        return list(self._advertised_tools)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> list[types.ContentBlock]:
        if name not in self._allowed_set:
            raise ToolNotAllowedError("Tool is not enabled for this task.")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("Tool arguments must be an object.")
        if not self._advertised_tools:
            await self.refresh_tools()
        response = await self.client.request_json(
            "POST",
            "/call-tool",
            {
                "tool_name": name,
                "tool_args": dict(arguments or {}),
                "use_cache": True,
            },
        )
        if not isinstance(response, list):
            raise AtlasBackendError("Atlas call-tool response must be an array.")
        return [_content_from_json(item) for item in response]

    async def check_ready(self) -> int:
        health = await self.client.request_json("GET", "/health")
        if not isinstance(health, Mapping) or health.get("status") != (
            "health_and_client_connection_ok"
        ):
            raise AtlasBackendError("Atlas backend health check failed.")
        return len(await self.refresh_tools())

    async def wait_until_ready(
        self,
        *,
        timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC,
        retry_interval_sec: float = DEFAULT_STARTUP_RETRY_INTERVAL_SEC,
    ) -> int:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        last_error: AtlasBackendError | None = None
        while loop.time() < deadline:
            try:
                return await self.check_ready()
            except AtlasBackendError as exc:
                last_error = exc
            await asyncio.sleep(min(retry_interval_sec, max(0, deadline - loop.time())))
        raise AtlasBackendError(
            "Atlas backend did not become ready in time."
        ) from last_error


class _MCPASGIApp:
    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        self.manager = manager

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.manager.handle_request(scope, receive, send)


def create_app(
    bridge: MCPAtlasBridge,
    *,
    startup_timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC,
) -> Starlette:
    server = Server("harbor-mcp-atlas-gateway", version="1.0.0")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return await bridge.list_tools()

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.ContentBlock]:
        return await bridge.call_tool(name, arguments)

    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=False
    )

    async def health(_: Request) -> JSONResponse:
        try:
            count = await bridge.check_ready()
        except AtlasBackendError:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok", "tool_count": count})

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        await bridge.wait_until_ready(timeout_sec=startup_timeout_sec)
        async with manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/mcp", endpoint=_MCPASGIApp(manager)),
        ],
        lifespan=lifespan,
    )
    app.state.mcp_server = server
    app.state.session_manager = manager
    app.state.bridge = bridge
    return app


def _positive_float(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise BridgeConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(result) or result <= 0:
        raise BridgeConfigurationError(f"{name} must be a positive finite number.")
    return result


def bridge_from_environment() -> tuple[MCPAtlasBridge, float]:
    allowlist = load_allowlist(
        Path(
            os.environ.get("MCP_ATLAS_ENABLED_TOOLS_FILE", str(DEFAULT_ALLOWLIST_PATH))
        )
    )
    timeout = _positive_float(
        os.environ.get("MCP_ATLAS_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)),
        "MCP_ATLAS_TIMEOUT_SEC",
    )
    startup_timeout = _positive_float(
        os.environ.get(
            "MCP_ATLAS_STARTUP_TIMEOUT_SEC", str(DEFAULT_STARTUP_TIMEOUT_SEC)
        ),
        "MCP_ATLAS_STARTUP_TIMEOUT_SEC",
    )
    client = AtlasRESTClient(
        os.environ.get("MCP_ATLAS_BASE_URL", DEFAULT_BASE_URL), timeout_sec=timeout
    )
    return MCPAtlasBridge(allowlist, client), startup_timeout


def main() -> None:
    bridge, startup_timeout = bridge_from_environment()
    app = create_app(bridge, startup_timeout_sec=startup_timeout)
    try:
        port = int(os.environ.get("MCP_BRIDGE_PORT", str(DEFAULT_PORT)))
    except ValueError as exc:
        raise BridgeConfigurationError("MCP_BRIDGE_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise BridgeConfigurationError("MCP_BRIDGE_PORT is out of range.")
    uvicorn.run(
        app,
        host=os.environ.get("MCP_BRIDGE_HOST", DEFAULT_HOST),
        port=port,
        access_log=False,
        log_level="info",
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
