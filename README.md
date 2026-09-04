# MCP-Atlas for Harbor

This repository is the immutable, generated [Harbor](https://github.com/Osmosis-AI/harbor) distribution of [Scale AI MCP-Atlas](https://github.com/scaleapi/mcp-atlas), a benchmark for tool-use competency with real Model Context Protocol (MCP) servers.

Release **v1.0.0** contains the complete pinned public split: **500 tasks**, **1,952 normalized claims**, a strict **30-task credential-free subset**, and a fixed **5-task smoke subset**. Reference trajectories are intentionally excluded.

## Datasets

| Dataset | Tasks | Intended use |
|---|---:|---|
| `mcp-atlas-smoke@1.0.0` | 5 | First deployment check; credential-free and intentionally small. |
| `mcp-atlas-credential-free@1.0.0` | 30 | Evaluation without Atlas service credentials. An evaluator model is still required. |
| `mcp-atlas@1.0.0` | 500 | Full public split; credentials and prepared external-service state may be required. |

Start with the smoke set on local Docker and one concurrent trial:

```bash
harbor run \
  --repo Osmosis-AI/mcp-atlas-harbor@v1.0.0 \
  --dataset mcp-atlas-smoke@1.0.0 \
  --agent <agent> \
  --model <model>
```

The verifier uses an OpenAI-compatible judge endpoint:

```bash
export EVAL_LLM_API_KEY='...'
export EVAL_LLM_BASE_URL='https://your-endpoint'
export EVAL_LLM_MODEL='your-judge-model'
```

Agent-model credentials are configured normally for the selected Harbor agent.

## Runtime and safety

This release officially supports **local Docker only**. MCP-Atlas is resource-heavy; allow at least 8 GB of memory and more than a minute for a cold start. Remote Compose providers and Podman have different startup, networking, mount-label, and secret-injection behavior and are not release-supported here.

Only 30 of 500 tasks have a completely credential-free original tool allowlist. The remaining 470 may need one or more Atlas service credentials and upstream-prepared external state. Export only the credentials required by the selected tasks in the host shell. Never pass Atlas service credentials through Harbor `--agent-env` / `--ae` or task environment variables, because those paths can expose values to the main agent container.

The generated topology isolates each credential-bearing server in its own sidecar, egress network, and Unix socket. The agent reaches only an allowlisting MCP gateway and receives neither backend sockets nor Atlas credentials. Use short-lived credentials for disposable accounts with minimum permissions.

**33 tasks expose known mutating tools**, including Slack posting and GitHub create/update operations. Inspect task metadata before running the full dataset, use disposable prepared accounts, and assume a successful tool call can change external state.

The official Atlas image is referenced by digest; this repository does not redistribute or publish a derivative container image.

## Evaluation fidelity

The adapter preserves each prompt, the available gold tools, the official MCP server implementations, and the official per-claim grades: fulfilled = 1, partially fulfilled = 0.5, and not fulfilled = 0. A task reward is the arithmetic mean rounded to three decimals. The verifier also reports pass indicators at coverage thresholds 0.50 and 0.75.

The public reference `TRAJECTORY` column is dropped before normalization and is never copied into a task. Ground-truth claims and the oracle answer live only in Harbor verifier/solution surfaces, not in the normal agent environment.

Four obsolete distractor prefixes in the public data—`anili`, `balldontlie`, `f1-mcp-server`, and `rijksmuseum-server`—are absent from the pinned official image. They occur in 85 tasks but are never called by the public reference trajectories. The adapter removes only those unavailable distractors and records the removal in task metadata.

Quantitative original-versus-Harbor model parity has not yet been run. Smoke and oracle validation establish packaging/runtime correctness, not model-score parity.

## Immutable provenance

| Input | Pin |
|---|---|
| Hugging Face dataset | `ScaleAI/MCP-Atlas` at `8c563b55d7c967755f474299848049834d624617` |
| Public Parquet | `MCP-Atlas.parquet`, 15,638,757 bytes, SHA-256 `2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0` |
| Upstream source | `scaleapi/mcp-atlas` at `f24ba3fb0bfa484c86acb28431fad6d7282455f9` |
| Harbor adapter | `Osmosis-AI/harbor` at `ae2f65b3657a4061bf85116cd381979b86d53fa4` |
| Official Atlas image | `ghcr.io/scaleapi/mcp-atlas:1.2.7@sha256:24e6ed3534916afe2c6825382da159a30e23516ef612be5d074fd96a74f9184c` |
| Main/verifier image | `python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254` |

`manifests/mcp-atlas-1.0.0.json` records those inputs, runtime Git pins, every task checksum, subset membership, mutation flags, and removed distractors. Its `dataset.git_commit` is intentionally null because embedding a commit hash in content changes that same commit; the annotated `v1.0.0` tag is the authoritative immutable repository pin.

## Repository layout

```text
.
├── tasks/                              # 500 generated Harbor task directories
├── manifests/mcp-atlas-1.0.0.json     # provenance and per-task checksums
├── registry.json                       # full, credential-free, and smoke views
├── scripts/release.sh                  # regenerate, validate, and byte-diff
└── .github/workflows/validate.yml      # reproducibility gate
```

No raw Parquet, source exports, credentials, trajectories, image archives, or generated evaluation outputs belong in this repository.

## Reproduce the release

The check script reads the adapter commit and dataset version from the committed manifest, checks out that exact Harbor revision, downloads and verifies the pinned source artifact, regenerates all 500 tasks, runs the adapter validator, and compares tasks, registry, and manifest byte-for-byte:

```bash
./scripts/release.sh --check
```

For an already verified local Harbor checkout at the exact manifest commit:

```bash
HARBOR_SRC=/path/to/harbor ./scripts/release.sh --check
```

You may set `MCP_ATLAS_SOURCE_FILE=/path/to/MCP-Atlas.parquet`; the adapter still verifies its pinned SHA-256. Published version tags are immutable and must never be moved or amended.

## Licensing and attribution

Benchmark-derived prompts, claims, answers, and task-specific data are licensed under **CC BY 4.0**; see [LICENSE-DATA](LICENSE-DATA). Harbor-authored code, scripts, runtime bridge, verifier, and repository documentation are licensed under **Apache-2.0**; see [LICENSE-CODE](LICENSE-CODE). Mixed generated files retain the applicable license for each contribution.

See [NOTICE](NOTICE) for attribution and modifications, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for referenced third-party software, and [CITATION.bib](CITATION.bib) for the benchmark citation.
