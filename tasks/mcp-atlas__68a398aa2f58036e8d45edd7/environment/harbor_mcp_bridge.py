#!/usr/bin/env python3
"""Expose isolated MCP-Atlas REST backends through one allowlisted MCP gateway."""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import math
import os
import socket
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, override

import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


DEFAULT_ALLOWLIST_PATH = Path("/harbor/enabled_tools.json")
DEFAULT_BACKENDS_PATH = Path("/harbor/atlas_backends.json")
DEFAULT_ALLOWED_CONFIG_ROOT = Path("/harbor")
DEFAULT_SOCKET_ROOT = Path("/run/mcp-atlas-backends")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18765
DEFAULT_TIMEOUT_SEC = 180.0
DEFAULT_STARTUP_TIMEOUT_SEC = 240.0
DEFAULT_STARTUP_RETRY_INTERVAL_SEC = 1.0
MAX_BACKEND_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
_SERVER_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


class BridgeError(RuntimeError):
    """Base class for fail-closed gateway errors."""


class BridgeConfigurationError(BridgeError):
    """The generated runtime configuration is invalid or unsafe."""


class AtlasBackendError(BridgeError):
    """A private Atlas REST backend failed or returned invalid data."""


class ToolNotAllowedError(BridgeError):
    """A caller attempted to bypass the task-local tool allowlist."""


def _valid_server_name(value: str) -> bool:
    return (
        bool(value)
        and value[0] in "abcdefghijklmnopqrstuvwxyz0123456789"
        and all(character in _SERVER_NAME_CHARS for character in value)
    )


def _trusted_file(path: Path, *, allowed_root: Path) -> Path:
    """Resolve a regular file without permitting symlink/path escape tricks."""

    if not path.is_absolute():
        raise BridgeConfigurationError("Runtime file paths must be absolute.")
    try:
        if path.is_symlink():
            raise BridgeConfigurationError("Runtime configuration cannot be a symlink.")
        resolved = path.resolve(strict=True)
        root = allowed_root.resolve(strict=True)
    except OSError as exc:
        raise BridgeConfigurationError(
            "Runtime configuration is not readable."
        ) from exc
    if not resolved.is_file() or resolved.parent != root:
        raise BridgeConfigurationError(
            "Runtime configuration escaped its trusted root."
        )
    return resolved


def _load_json_file(path: Path, *, allowed_root: Path) -> Any:
    trusted = _trusted_file(path, allowed_root=allowed_root)
    try:
        if trusted.stat().st_size > MAX_CONFIG_BYTES:
            raise BridgeConfigurationError(
                "Runtime configuration is unexpectedly large."
            )
        return json.loads(trusted.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeConfigurationError(
            "Runtime configuration is not valid UTF-8 JSON."
        ) from exc


def load_allowlist(
    path: Path = DEFAULT_ALLOWLIST_PATH,
    *,
    allowed_root: Path = DEFAULT_ALLOWED_CONFIG_ROOT,
) -> tuple[str, ...]:
    """Load a non-empty, duplicate-free JSON array of external tool names."""

    value = _load_json_file(path, allowed_root=allowed_root)
    if not isinstance(value, list) or not value:
        raise BridgeConfigurationError("Tool allowlist must be a non-empty JSON array.")
    tools: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item or item.strip() != item:
            raise BridgeConfigurationError(
                "Every allowlist entry must be a non-empty, trimmed string."
            )
        if any(character.isspace() or ord(character) < 0x21 for character in item):
            raise BridgeConfigurationError("Tool names cannot contain whitespace.")
        if item in seen:
            raise BridgeConfigurationError("Tool allowlist contains a duplicate name.")
        seen.add(item)
        tools.append(item)
    return tuple(tools)


def _validated_socket_path(value: str, *, socket_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.name != "atlas.sock":
        raise BridgeConfigurationError("Atlas backend socket path is invalid.")
    try:
        relative = path.relative_to(socket_root)
    except ValueError as exc:
        raise BridgeConfigurationError(
            "Atlas backend socket escaped its trusted root."
        ) from exc
    if len(relative.parts) != 2 or not _valid_server_name(relative.parts[0]):
        raise BridgeConfigurationError("Atlas backend socket path is invalid.")
    return path


def load_backend_map(
    path: Path = DEFAULT_BACKENDS_PATH,
    *,
    allowed_root: Path = DEFAULT_ALLOWED_CONFIG_ROOT,
    socket_root: Path = DEFAULT_SOCKET_ROOT,
) -> dict[str, Path]:
    """Load the exact server-to-private-socket routing table."""

    value = _load_json_file(path, allowed_root=allowed_root)
    if not isinstance(value, dict) or not value:
        raise BridgeConfigurationError("Atlas backend map must be a non-empty object.")
    result: dict[str, Path] = {}
    for server, raw_socket in value.items():
        if not isinstance(server, str) or not _valid_server_name(server):
            raise BridgeConfigurationError(
                "Atlas backend map has an invalid server name."
            )
        if not isinstance(raw_socket, str):
            raise BridgeConfigurationError("Atlas backend socket must be a string.")
        result[server] = _validated_socket_path(raw_socket, socket_root=socket_root)
    return result


class UnixHTTPConnection(http.client.HTTPConnection):
    """A fresh HTTP/1.1 connection transported over an AF_UNIX socket."""

    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = str(socket_path)

    @override
    def connect(self) -> None:
        try:
            mode = os.lstat(self.socket_path).st_mode
        except OSError as exc:
            raise AtlasBackendError("Atlas backend socket is unavailable.") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISSOCK(mode):
            raise AtlasBackendError("Atlas backend socket is not a Unix socket.")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(self.socket_path)
        except BaseException:
            connection.close()
            raise
        self.sock = connection


ConnectionFactory = Callable[[Path, float], http.client.HTTPConnection]


class AtlasUnixClient:
    """Small async facade over one private Atlas REST API."""

    _ALLOWED_ENDPOINTS = frozenset({"/call-tool", "/health", "/list-tools"})

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_response_bytes: int = MAX_BACKEND_RESPONSE_BYTES,
        connection_factory: ConnectionFactory = UnixHTTPConnection,
    ) -> None:
        if not socket_path.is_absolute() or socket_path.name != "atlas.sock":
            raise BridgeConfigurationError("Atlas socket path is invalid.")
        if not math.isfinite(timeout_sec):
            raise BridgeConfigurationError("Atlas client timeout must be finite.")
        if timeout_sec <= 0 or max_response_bytes < 1:
            raise BridgeConfigurationError("Atlas client limits must be positive.")
        self.socket_path = socket_path
        self.timeout_sec = timeout_sec
        self.max_response_bytes = max_response_bytes
        self.connection_factory = connection_factory

    async def request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._request_json, method, endpoint, payload)

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if endpoint not in self._ALLOWED_ENDPOINTS:
            raise BridgeConfigurationError("Atlas REST endpoint is not allowed.")
        if method not in {"GET", "POST"}:
            raise BridgeConfigurationError("Atlas REST method is not allowed.")
        body: bytes | None = None
        headers = {"Accept": "application/json", "Connection": "close"}
        if payload is not None:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection = self.connection_factory(self.socket_path, self.timeout_sec)
        try:
            connection.request(method, endpoint, body=body, headers=headers)
            response = connection.getresponse()
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise AtlasBackendError(
                        "Atlas backend returned an invalid response length."
                    ) from exc
                if declared_size > self.max_response_bytes:
                    raise AtlasBackendError(
                        "Atlas backend response exceeded the safe limit."
                    )
            raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise AtlasBackendError(
                    "Atlas backend response exceeded the safe limit."
                )
            if not 200 <= response.status < 300:
                # A backend error body can contain subprocess arguments and secrets.
                raise AtlasBackendError(
                    f"Atlas backend request failed with HTTP {response.status}."
                )
            try:
                return json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AtlasBackendError("Atlas backend returned invalid JSON.") from exc
        except AtlasBackendError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise AtlasBackendError("Atlas backend is unavailable.") from exc
        finally:
            connection.close()


def _tool_from_json(value: Any, *, advertised_name: str) -> types.Tool:
    if not isinstance(value, Mapping):
        raise AtlasBackendError("Atlas list-tools response contains a non-object.")
    if not isinstance(value.get("name"), str) or not isinstance(
        value.get("inputSchema"), dict
    ):
        raise AtlasBackendError("Atlas list-tools response has an invalid tool.")
    sanitized = dict(value)
    sanitized["name"] = advertised_name
    # Atlas's REST facade returns only MCP content blocks from /call-tool and
    # discards structuredContent. Advertising the backend outputSchema would
    # therefore make the MCP SDK reject otherwise successful tool results.
    sanitized.pop("outputSchema", None)
    try:
        return types.Tool.model_validate(sanitized)
    except Exception as exc:
        raise AtlasBackendError(
            "Atlas list-tools response has an invalid tool."
        ) from exc


def _content_from_json(value: Any) -> types.ContentBlock:
    if not isinstance(value, Mapping):
        raise AtlasBackendError("Atlas call-tool response contains a non-object.")
    content_type = {
        "audio": getattr(types, "AudioContent", None),
        "image": types.ImageContent,
        "resource": types.EmbeddedResource,
        "resource_link": getattr(types, "ResourceLink", None),
        "text": types.TextContent,
    }.get(value.get("type"))
    if content_type is None:
        raise AtlasBackendError("Atlas call-tool response has an unknown content type.")
    try:
        return content_type.model_validate(value)
    except Exception as exc:
        raise AtlasBackendError(
            "Atlas call-tool response has invalid content."
        ) from exc


ClientFactory = Callable[[Path], AtlasUnixClient]


class MCPAtlasBridge:
    """Route an exact external allowlist across isolated Atlas backends."""

    def __init__(
        self,
        allowed_tools: Sequence[str],
        backend_sockets: Mapping[str, Path],
        *,
        client_factory: ClientFactory,
    ) -> None:
        self.allowed_tools = tuple(allowed_tools)
        self._allowed_set = frozenset(allowed_tools)
        if not self.allowed_tools or len(self.allowed_tools) != len(self._allowed_set):
            raise BridgeConfigurationError(
                "Allowed tools must be non-empty and unique."
            )
        if not backend_sockets:
            raise BridgeConfigurationError("At least one Atlas backend is required.")
        self.backend_sockets = dict(backend_sockets)
        self._tool_servers = {
            tool: self._server_for_tool(tool) for tool in self.allowed_tools
        }
        if set(self._tool_servers.values()) != set(self.backend_sockets):
            raise BridgeConfigurationError(
                "Atlas backend map must exactly cover allowlisted tool servers."
            )
        self._servers_by_socket: dict[Path, tuple[str, ...]] = {}
        for socket_path in sorted(set(self.backend_sockets.values()), key=str):
            self._servers_by_socket[socket_path] = tuple(
                sorted(
                    server
                    for server, configured_socket in self.backend_sockets.items()
                    if configured_socket == socket_path
                )
            )
        self._clients = {
            socket_path: client_factory(socket_path)
            for socket_path in self._servers_by_socket
        }
        self._tool_mapping: dict[str, tuple[AtlasUnixClient, str]] = {}
        self._advertised_tools: tuple[types.Tool, ...] = ()
        self._refresh_lock = asyncio.Lock()
        self._ready = False
        self._ready_tool_count = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def ready_tool_count(self) -> int:
        return self._ready_tool_count

    def _server_for_tool(self, tool: str) -> str:
        matches = [
            server for server in self.backend_sockets if tool.startswith(f"{server}_")
        ]
        if len(matches) != 1:
            raise BridgeConfigurationError(
                "Every allowlisted tool must have exactly one configured server prefix."
            )
        return matches[0]

    async def _query_backend(
        self, socket_path: Path, *, check_health: bool
    ) -> tuple[Path, Any]:
        client = self._clients[socket_path]
        try:
            if check_health:
                health = await client.request_json("GET", "/health")
                if not isinstance(health, Mapping) or health.get("status") != (
                    "health_and_client_connection_ok"
                ):
                    raise AtlasBackendError("Atlas backend health check failed.")
            tools = await client.request_json("POST", "/list-tools")
        except Exception as exc:
            # Collapse all backend detail before it can reach an agent-facing route.
            raise AtlasBackendError("Atlas backend is unavailable.") from exc
        return socket_path, tools

    async def _collect_backend_tools(self, *, check_health: bool) -> dict[Path, Any]:
        tasks: dict[Path, asyncio.Task[tuple[Path, Any]]] = {}
        try:
            async with asyncio.TaskGroup() as group:
                for socket_path in self._clients:
                    tasks[socket_path] = group.create_task(
                        self._query_backend(socket_path, check_health=check_health)
                    )
        except* Exception as group_error:
            raise AtlasBackendError(
                "One or more Atlas backends are unavailable."
            ) from (group_error)
        return {socket_path: task.result()[1] for socket_path, task in tasks.items()}

    def _resolve_tool_mapping(
        self, backend_tools: Mapping[Path, Any]
    ) -> tuple[dict[str, tuple[AtlasUnixClient, str]], tuple[types.Tool, ...]]:
        mapping: dict[str, tuple[AtlasUnixClient, str]] = {}
        advertised_by_name: dict[str, types.Tool] = {}
        for socket_path, raw_tools in backend_tools.items():
            if not isinstance(raw_tools, list):
                raise AtlasBackendError("Atlas list-tools response must be an array.")
            by_name: dict[str, Any] = {}
            for raw_tool in raw_tools:
                if not isinstance(raw_tool, Mapping) or not isinstance(
                    raw_tool.get("name"), str
                ):
                    raise AtlasBackendError(
                        "Atlas list-tools response has an invalid tool."
                    )
                actual_name = raw_tool["name"]
                if actual_name in by_name:
                    raise AtlasBackendError(
                        "Atlas list-tools response has duplicate names."
                    )
                by_name[actual_name] = raw_tool

            servers = self._servers_by_socket[socket_path]
            external_tools = [
                tool
                for tool in self.allowed_tools
                if self.backend_sockets[self._tool_servers[tool]] == socket_path
            ]
            used_actual_names: set[str] = set()
            for external_name in external_tools:
                actual_name = external_name
                if actual_name not in by_name and len(servers) == 1:
                    prefix = f"{servers[0]}_"
                    if external_name.startswith(prefix):
                        stripped = external_name[len(prefix) :]
                        if stripped in by_name:
                            actual_name = stripped
                if actual_name not in by_name:
                    raise AtlasBackendError(
                        "Atlas backend is missing one or more allowlisted tools."
                    )
                if actual_name in used_actual_names:
                    raise AtlasBackendError("Atlas tool-name adaptation is ambiguous.")
                used_actual_names.add(actual_name)
                mapping[external_name] = (self._clients[socket_path], actual_name)
                advertised_by_name[external_name] = _tool_from_json(
                    by_name[actual_name], advertised_name=external_name
                )
        if set(mapping) != self._allowed_set:
            raise AtlasBackendError("Atlas backend tool coverage is incomplete.")
        return mapping, tuple(advertised_by_name[name] for name in self.allowed_tools)

    async def refresh_tools(
        self, *, check_health: bool = False
    ) -> tuple[types.Tool, ...]:
        async with self._refresh_lock:
            backend_tools = await self._collect_backend_tools(check_health=check_health)
            mapping, tools = self._resolve_tool_mapping(backend_tools)
            self._tool_mapping = mapping
            self._advertised_tools = tools
            return tools

    async def list_tools(self) -> list[types.Tool]:
        tools = self._advertised_tools or await self.refresh_tools()
        return list(tools)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> list[types.ContentBlock]:
        if name not in self._allowed_set:
            raise ToolNotAllowedError("Tool is not enabled for this task.")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise BridgeError("Tool arguments must be an object.")
        if name not in self._tool_mapping:
            await self.refresh_tools()
        route = self._tool_mapping.get(name)
        if route is None:
            raise ToolNotAllowedError("Tool is not enabled for this task.")
        client, actual_name = route
        response = await client.request_json(
            "POST",
            "/call-tool",
            {
                "tool_name": actual_name,
                "tool_args": dict(arguments or {}),
                "use_cache": True,
            },
        )
        if not isinstance(response, list):
            raise AtlasBackendError("Atlas call-tool response must be an array.")
        if not response:
            return [types.TextContent(type="text", text="success")]
        return [_content_from_json(item) for item in response]

    async def check_ready(self) -> int:
        self._ready = False
        tools = await self.refresh_tools(check_health=True)
        self._ready_tool_count = len(tools)
        self._ready = True
        return self._ready_tool_count

    async def wait_until_ready(
        self,
        *,
        timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC,
        retry_interval_sec: float = DEFAULT_STARTUP_RETRY_INTERVAL_SEC,
    ) -> int:
        """Wait for concurrently starting backends without leaking failure detail."""

        if not math.isfinite(timeout_sec) or not math.isfinite(retry_interval_sec):
            raise BridgeConfigurationError("Startup timing values must be finite.")
        if timeout_sec <= 0 or retry_interval_sec <= 0:
            raise BridgeConfigurationError("Startup timing values must be positive.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        last_error: AtlasBackendError | None = None
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    return await self.check_ready()
            except TimeoutError:
                break
            except AtlasBackendError as exc:
                last_error = exc
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(retry_interval_sec, remaining))
        raise AtlasBackendError(
            "Atlas backends did not become ready before the startup deadline."
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
    """Create a stateful streamable-HTTP MCP app with redacted health errors."""

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
        app=server,
        json_response=True,
        stateless=False,
    )

    async def health(_: Request) -> JSONResponse:
        if not bridge.ready:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok", "tool_count": bridge.ready_tool_count})

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


def _positive_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise BridgeConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise BridgeConfigurationError(f"{name} must be finite.")
    if parsed <= 0:
        raise BridgeConfigurationError(f"{name} must be positive.")
    return parsed


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BridgeConfigurationError("MCP_BRIDGE_PORT must be an integer.") from exc
    if not 1 <= parsed <= 65535:
        raise BridgeConfigurationError("MCP_BRIDGE_PORT is out of range.")
    return parsed


def main() -> None:
    allowlist = load_allowlist(
        Path(
            os.environ.get("MCP_ATLAS_ENABLED_TOOLS_FILE", str(DEFAULT_ALLOWLIST_PATH))
        )
    )
    backend_sockets = load_backend_map(
        Path(os.environ.get("MCP_ATLAS_BACKENDS_FILE", str(DEFAULT_BACKENDS_PATH)))
    )
    timeout = _positive_float(
        os.environ.get("MCP_ATLAS_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)),
        name="MCP_ATLAS_TIMEOUT_SEC",
    )
    startup_timeout = _positive_float(
        os.environ.get(
            "MCP_ATLAS_STARTUP_TIMEOUT_SEC", str(DEFAULT_STARTUP_TIMEOUT_SEC)
        ),
        name="MCP_ATLAS_STARTUP_TIMEOUT_SEC",
    )
    bridge = MCPAtlasBridge(
        allowlist,
        backend_sockets,
        client_factory=lambda path: AtlasUnixClient(path, timeout_sec=timeout),
    )
    app = create_app(bridge, startup_timeout_sec=startup_timeout)
    uvicorn.run(
        app,
        host=os.environ.get("MCP_BRIDGE_HOST", DEFAULT_HOST),
        port=_port(os.environ.get("MCP_BRIDGE_PORT", str(DEFAULT_PORT))),
        access_log=False,
        log_level="info",
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
