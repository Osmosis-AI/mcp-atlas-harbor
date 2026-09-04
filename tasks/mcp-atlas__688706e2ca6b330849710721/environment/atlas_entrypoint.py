#!/usr/bin/env python3
"""Prepare the pinned MCP-Atlas image and serve it on a private Unix socket.

This replaces the image's shell ``envsubst`` entrypoint.  It filters disabled
servers before substituting secrets, writes the materialized config with mode
0600, and redirects exact ``uvx`` package specs to executables baked into the
image.  Remaining source-based uvx commands inherit an ``mcp<2`` constraint.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override


AGENT_ENVIRONMENT_ROOT = Path("/agent-environment")
TEMPLATE_PATH = (
    AGENT_ENVIRONMENT_ROOT / "src" / "agent_environment" / "mcp_server_template.json"
)
CONFIG_PATH = (
    AGENT_ENVIRONMENT_ROOT / "src" / "agent_environment" / "mcp_server_config.json"
)
SOCKET_ROOT = Path("/run/mcp-atlas")
SOCKET_PATH = SOCKET_ROOT / "atlas.sock"
PYTHON = AGENT_ENVIRONMENT_ROOT / ".venv" / "bin" / "python"
CONSTRAINT_PATH = Path("/harbor/uv-constraints.txt")
BIN_DIR = Path("/usr/local/bin")
_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_SERVER_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\Z")

# install_mcp_packages.sh in the pinned 1.2.7 image installs these exact tools
# during the image build.  Calling uvx again needlessly re-resolves their
# dependency graphs and can pull an incompatible MCP major version.
UVX_EXECUTABLES: dict[str, str] = {
    "arxiv-mcp-server==0.2.11": "arxiv-mcp-server",
    "cli-mcp-server==0.2.5": "cli-mcp-server",
    "duckduckgo-mcp-server==0.5.0": "duckduckgo-mcp-server",
    "mcp-server-calculator==0.2.0": "mcp-server-calculator",
    "mcp-server-fetch==2025.4.7": "mcp-server-fetch",
    "mcp-server-git==2026.7.10": "mcp-server-git",
    "mcp-server-twelve-data==0.2.5": "mcp-server-twelve-data",
    "osm-mcp-server==0.1.1": "osm-mcp-server",
    "oxylabs-mcp==0.4.1": "oxylabs-mcp",
    "wikipedia-mcp==2.0.1": "wikipedia-mcp",
}

# The pinned image template contains four GitHub default-branch references.
# Rewrite only those exact known values; any other Git source fails closed.
GIT_SOURCE_REWRITES: dict[str, str] = {
    "https://github.com/geobio/smitheryai-mcp-servers-github": (
        "https://github.com/geobio/smitheryai-mcp-servers-github"
        "#68368436034fb0003a6d8ed91afc9d0a64142b84"
    ),
    "https://github.com/geobio/smitheryai-mcp-servers-weather": (
        "https://github.com/geobio/smitheryai-mcp-servers-weather"
        "#3474c7841d00f5f40c087c5d2188a5c0f41bd134"
    ),
    "git+https://github.com/geobio/PubMed-MCP-Server.git": (
        "git+https://github.com/geobio/PubMed-MCP-Server.git"
        "@e452cfc7d23cd4c0248c177d7814df3d9e82ea3b"
    ),
    "git+https://github.com/geobio/weather-mcp-server": (
        "git+https://github.com/geobio/weather-mcp-server"
        "@70b6a7c8183c8a3acc175f786f4b7be9c2ba66e4"
    ),
}
PINNED_GIT_SOURCES = frozenset(GIT_SOURCE_REWRITES.values())


class EntrypointError(RuntimeError):
    """The pinned image or generated task violates the runtime contract."""


def parse_enabled_servers(value: str) -> tuple[str, ...]:
    servers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not servers:
        raise EntrypointError("ENABLED_SERVERS must select at least one server.")
    if len(set(servers)) != len(servers):
        raise EntrypointError("ENABLED_SERVERS contains duplicates.")
    if any(_SERVER_NAME.fullmatch(server) is None for server in servers):
        raise EntrypointError("ENABLED_SERVERS contains an invalid server name.")
    return servers


def parse_required_environment(value: str) -> tuple[str, ...]:
    """Parse the generated declaration of secrets owned by this backend."""

    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(set(names)) != len(names):
        raise EntrypointError("MCP_ATLAS_REQUIRED_ENV contains duplicates.")
    if any(re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None for name in names):
        raise EntrypointError("MCP_ATLAS_REQUIRED_ENV contains an invalid name.")
    return names


def _required_value(environment: Mapping[str, str], name: str) -> str:
    try:
        return environment[name]
    except KeyError as exc:
        raise EntrypointError(f"Required runtime setting {name} is missing.") from exc


def _environment_references(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_ENV_REFERENCE.findall(value))
    if isinstance(value, list):
        return set().union(*(_environment_references(item) for item in value), set())
    if isinstance(value, Mapping):
        return set().union(
            *(_environment_references(item) for item in value.values()), set()
        )
    return set()


def _substitute_environment(value: Any, environment: Mapping[str, str]) -> Any:
    """Substitute exact ${VAR} references while preserving JSON escaping."""

    if isinstance(value, str):
        return _ENV_REFERENCE.sub(lambda match: environment[match.group(1)], value)
    if isinstance(value, list):
        return [_substitute_environment(item, environment) for item in value]
    if isinstance(value, dict):
        return {
            key: _substitute_environment(item, environment)
            for key, item in value.items()
        }
    return value


def select_and_expand_config(
    template: Mapping[str, Any],
    enabled_servers: Sequence[str],
    environment: Mapping[str, str],
    declared_required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Drop disabled servers before expanding only the selected credentials."""

    raw_servers = template.get("mcpServers")
    if not isinstance(raw_servers, Mapping):
        raise EntrypointError("Atlas template has no mcpServers object.")
    unknown = sorted(set(enabled_servers) - set(raw_servers))
    if unknown:
        raise EntrypointError(
            "ENABLED_SERVERS names a server absent from the pinned Atlas image."
        )

    selected = {name: raw_servers[name] for name in enabled_servers}
    required = _environment_references(selected)
    if declared_required is not None and required != set(declared_required):
        raise EntrypointError(
            "Declared Atlas environment variables do not match the selected servers."
        )
    missing = sorted(name for name in required if not environment.get(name, "").strip())
    if missing:
        # Variable names are safe operational metadata; values are never logged.
        raise EntrypointError(
            "Missing required Atlas environment variables: " + ", ".join(missing)
        )
    expanded = _substitute_environment(selected, environment)
    return {"mcpServers": expanded}


def pin_git_sources(config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the image template's exact Git references to audited commits."""

    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise EntrypointError("Atlas runtime config has no mcpServers object.")
    for server_config in servers.values():
        if not isinstance(server_config, dict):
            raise EntrypointError("Atlas server configuration is invalid.")
        args = server_config.get("args")
        if args is None:
            continue
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise EntrypointError("Atlas server arguments are invalid.")
        server_config["args"] = [GIT_SOURCE_REWRITES.get(arg, arg) for arg in args]
        for arg in server_config["args"]:
            if "github.com/" in arg and arg not in PINNED_GIT_SOURCES:
                raise EntrypointError("Atlas config contains an unpinned Git source.")
    return config


def _baked_executable(
    package: str,
    *,
    bin_dir: Path,
    require_executable: bool,
) -> str:
    executable_name = UVX_EXECUTABLES.get(package)
    if executable_name is None:
        raise EntrypointError(
            "Pinned Atlas config contains an unknown exact uvx package."
        )
    executable = bin_dir / executable_name
    if require_executable and (
        not executable.is_file() or not os.access(executable, os.X_OK)
    ):
        raise EntrypointError(
            "A uv tool expected in the pinned Atlas image is unavailable."
        )
    return str(executable)


def patch_uvx_commands(
    config: dict[str, Any],
    *,
    bin_dir: Path = BIN_DIR,
    require_executable: bool = True,
) -> dict[str, Any]:
    """Use build-time uv tool shims for every exact package invocation."""

    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise EntrypointError("Atlas runtime config has no mcpServers object.")
    for server_config in servers.values():
        if not isinstance(server_config, dict) or server_config.get("command") != "uvx":
            continue
        args = server_config.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise EntrypointError("Atlas uvx server has invalid arguments.")

        package: str | None = None
        remaining: list[str] = []
        if args and "==" in args[0]:
            package = args[0]
            remaining = args[1:]
        elif len(args) >= 3 and args[0] == "--from" and "==" in args[1]:
            package = args[1]
            expected_command = UVX_EXECUTABLES.get(package)
            if expected_command is None or args[2] != expected_command:
                raise EntrypointError(
                    "Atlas --from uvx command is not pinned as expected."
                )
            remaining = args[3:]

        if package is not None:
            server_config["command"] = _baked_executable(
                package,
                bin_dir=bin_dir,
                require_executable=require_executable,
            )
            server_config["args"] = remaining
    return config


def _read_template(path: Path = TEMPLATE_PATH) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise EntrypointError("Atlas template path is unsafe.")
    try:
        resolved = path.resolve(strict=True)
        root = AGENT_ENVIRONMENT_ROOT.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EntrypointError("Cannot read the pinned Atlas template.") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EntrypointError("Atlas template escaped the image root.") from exc
    if not isinstance(value, dict):
        raise EntrypointError("Atlas template must contain a JSON object.")
    return value


def write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically materialize credential-bearing JSON with owner-only access."""

    if not path.is_absolute() or path.is_symlink():
        raise EntrypointError("Atlas config path is unsafe.")
    parent = path.parent.resolve(strict=True)
    root = AGENT_ENVIRONMENT_ROOT.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise EntrypointError("Atlas config escaped the image root.") from exc

    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=parent
        )
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = ""
        os.chmod(path, 0o600)
    except OSError as exc:
        raise EntrypointError("Cannot write the private Atlas config.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def prepare_socket_path(
    socket_path: Path = SOCKET_PATH, socket_root: Path = SOCKET_ROOT
) -> None:
    if not socket_path.is_absolute() or socket_path.parent != socket_root:
        raise EntrypointError("Atlas Unix socket path is unsafe.")
    try:
        socket_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        os.chmod(socket_root, 0o700)
        if socket_path.is_symlink():
            raise EntrypointError("Atlas Unix socket cannot be a symlink.")
        if socket_path.exists():
            mode = socket_path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISSOCK(mode):
                raise EntrypointError("Refusing to replace a non-socket runtime file.")
            socket_path.unlink()
    except OSError as exc:
        raise EntrypointError("Cannot prepare the Atlas Unix socket.") from exc


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = str(socket_path)

    @override
    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(self.socket_path)
        except BaseException:
            connection.close()
            raise
        self.sock = connection


def check_health(socket_path: Path = SOCKET_PATH, timeout_sec: float = 5.0) -> bool:
    connection = _UnixHTTPConnection(socket_path, timeout_sec)
    try:
        connection.request(
            "GET",
            "/",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        body = response.read(64 * 1024)
        if response.status != 200:
            return False
        value = json.loads(body)
        return isinstance(value, dict) and value.get("message") == (
            "MCP Agent Environment API"
        )
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError):
        return False
    finally:
        connection.close()


def _configured_socket(environment: Mapping[str, str]) -> Path:
    socket_path = Path(_required_value(environment, "MCP_ATLAS_SOCKET"))
    if socket_path != SOCKET_PATH:
        raise EntrypointError("MCP_ATLAS_SOCKET must use the private mounted path.")
    return socket_path


def prepare_runtime(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    enabled_servers = parse_enabled_servers(_required_value(values, "ENABLED_SERVERS"))
    required_environment = parse_required_environment(
        _required_value(values, "MCP_ATLAS_REQUIRED_ENV")
    )
    socket_path = _configured_socket(values)
    constraint_path = _required_value(values, "UV_CONSTRAINT")
    if constraint_path != str(CONSTRAINT_PATH):
        raise EntrypointError("UV_CONSTRAINT must use the mounted constraints file.")
    template = _read_template()
    config = select_and_expand_config(
        template,
        enabled_servers,
        values,
        declared_required=required_environment,
    )
    pin_git_sources(config)
    patch_uvx_commands(config)
    write_private_json(CONFIG_PATH, config)
    prepare_socket_path(socket_path, socket_path.parent)

    # Constrain source-based uvx fallbacks (PubMed and weather-data). Exact
    # packages above never resolve at runtime.
    os.environ["UV_CONSTRAINT"] = str(CONSTRAINT_PATH)
    return socket_path


def serve() -> None:
    socket_path = prepare_runtime()
    os.chdir(AGENT_ENVIRONMENT_ROOT)
    os.umask(0o077)
    arguments = [
        str(PYTHON),
        "-m",
        "uvicorn",
        "agent_environment.main:app",
        "--uds",
        str(socket_path),
        "--no-access-log",
        "--no-proxy-headers",
    ]
    os.execv(str(PYTHON), arguments)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the private Atlas REST health endpoint and exit",
    )
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check_health(_configured_socket(os.environ)) else 1
    serve()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EntrypointError as exc:
        print(f"mcp-atlas entrypoint: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
