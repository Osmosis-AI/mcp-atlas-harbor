#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--check" || $# -ne 1 ]]; then
  echo "usage: $0 --check" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

shopt -s nullglob
manifest_files=("$repo_root"/manifests/mcp-atlas-*.json)
if [[ ${#manifest_files[@]} -ne 1 ]]; then
  echo "expected exactly one MCP-Atlas release manifest" >&2
  exit 1
fi
manifest="${manifest_files[0]}"

mapfile -t release_meta < <(
  python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["dataset"]["version"])
print(data["adapter"]["commit"])
print(data["source"]["revision"])
print(data["source"]["parquet_sha256"])
PY
)

if [[ ${#release_meta[@]} -ne 4 ]]; then
  echo "could not read release pins from $manifest" >&2
  exit 1
fi
dataset_version="${release_meta[0]}"
adapter_commit="${release_meta[1]}"
source_revision="${release_meta[2]}"
source_sha256="${release_meta[3]}"

if [[ ! "$dataset_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid dataset version: $dataset_version" >&2
  exit 1
fi
if [[ ! "$adapter_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid adapter commit: $adapter_commit" >&2
  exit 1
fi
if [[ ! "$source_revision" =~ ^[0-9a-f]{40}$ || ! "$source_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid source provenance in manifest" >&2
  exit 1
fi

for command in git python3 uv diff cmp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is missing: $command" >&2
    exit 1
  fi
done

python3 - "$repo_root" <<'PY'
import fnmatch
import sys
from pathlib import Path

root = Path(sys.argv[1])
patterns = (
    "*.parquet", "*.arrow", "*.feather", "*.csv", "*.tsv",
    "*trajectory*", ".env", ".env.*", "*credentials*.json",
    "*service-account*.json", "*.pem", "*.key", "*.tar", "*.tar.gz",
    "*.tgz", "*.zip", "*.sif",
)
forbidden = sorted(
    path.relative_to(root)
    for path in root.rglob("*")
    if path.is_file()
    and ".git" not in path.relative_to(root).parts
    and any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in patterns)
)
if forbidden:
    print("forbidden release artifacts found:", file=sys.stderr)
    for path in forbidden:
        print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)
PY

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

if [[ -n "${HARBOR_SRC:-}" ]]; then
  harbor_src="$(cd "$HARBOR_SRC" && pwd)"
  actual_commit="$(git -C "$harbor_src" rev-parse HEAD)"
  if [[ "$actual_commit" != "$adapter_commit" ]]; then
    echo "HARBOR_SRC is at $actual_commit, expected $adapter_commit" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$harbor_src" status --porcelain -- adapters/mcp-atlas)" ]]; then
    echo "HARBOR_SRC has uncommitted changes under adapters/mcp-atlas" >&2
    exit 1
  fi
else
  harbor_src="$work_dir/harbor"
  git init --quiet "$harbor_src"
  git -C "$harbor_src" remote add origin https://github.com/Osmosis-AI/harbor.git
  git -C "$harbor_src" fetch --quiet --depth 1 origin "$adapter_commit"
  git -C "$harbor_src" checkout --quiet --detach FETCH_HEAD
fi

adapter_dir="$harbor_src/adapters/mcp-atlas"
if [[ ! -f "$adapter_dir/pyproject.toml" ]]; then
  echo "pinned Harbor revision does not contain the MCP-Atlas adapter" >&2
  exit 1
fi

generated="$work_dir/generated"
source_args=()
if [[ -n "${MCP_ATLAS_SOURCE_FILE:-}" ]]; then
  source_args=(--source-file "$MCP_ATLAS_SOURCE_FILE")
fi

uv run --frozen --project "$adapter_dir" mcp-atlas generate \
  --repo-layout \
  --dataset-version "$dataset_version" \
  --adapter-commit "$adapter_commit" \
  --output-dir "$generated" \
  "${source_args[@]}"

uv run --frozen --project "$adapter_dir" mcp-atlas validate \
  --dataset-dir "$generated"

diff --no-dereference --recursive "$generated/tasks" "$repo_root/tasks"
cmp "$generated/registry.json" "$repo_root/registry.json"
cmp "$generated/manifests/mcp-atlas-$dataset_version.json" "$manifest"

echo "Release $dataset_version is reproducible from adapter $adapter_commit."
