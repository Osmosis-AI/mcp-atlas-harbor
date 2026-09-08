#!/usr/bin/env python3
"""Pre-install the enabled MCP servers, then start the MCP-Atlas runtime.

The runtime's /health probe connects to every enabled server with a five second
budget and cancels on timeout. Each connect spawns the servers with uvx or npx,
so on a cold container the first probe triggers first-time installs that take
longer than five seconds, get cancelled, and are retried and cancelled again
while compose keeps probing. Running every enabled server command once to
completion here (stdin closed, so stdio servers exit on EOF) fills the uv and
npm caches before the first probe. Cached starts take about a second.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TEMPLATE = "/agent-environment/src/agent_environment/mcp_server_template.json"
ENTRYPOINT = "/agent-environment/entrypoint.sh"
PER_SERVER_TIMEOUT_SEC = 240


def warm(name: str, spec: dict[str, Any]) -> str:
    command = [spec["command"], *spec.get("args", [])]
    env = dict(os.environ)
    env.update({k: v for k, v in spec.get("env", {}).items() if "${" not in v})
    try:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=PER_SERVER_TIMEOUT_SEC,
        )
        return f"[warmup] {name}: installed"
    except subprocess.TimeoutExpired:
        return f"[warmup] {name}: still running after {PER_SERVER_TIMEOUT_SEC}s, killed"
    except OSError as exc:
        return f"[warmup] {name}: {exc}"


def main() -> None:
    enabled = [s.strip() for s in os.environ.get("ENABLED_SERVERS", "").split(",")]
    template = json.loads(Path(TEMPLATE).read_text())
    servers = template["mcpServers"]
    if "slack" in enabled:
        # The pinned Slack server supports user OAuth, but the image template
        # only forwards browser session tokens. Keep authorization scoped to
        # the dedicated workspace without depending on a browser session.
        servers["slack"]["env"] = {"SLACK_MCP_XOXP_TOKEN": "${SLACK_MCP_XOXP_TOKEN}"}
        Path(TEMPLATE).write_text(json.dumps(template))
    targets = {name: servers[name] for name in enabled if name in servers}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(lambda item: warm(*item), targets.items()):
            print(line, flush=True)
    os.execv(ENTRYPOINT, [ENTRYPOINT, *sys.argv[1:]])


if __name__ == "__main__":
    main()
