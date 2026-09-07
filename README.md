# MCP-Atlas for Harbor

This repository contains the generated Harbor distribution of
[Scale AI MCP-Atlas](https://github.com/scaleapi/mcp-atlas), a benchmark for
tool use with real Model Context Protocol servers.

Release **v1.0.5** contains the complete pinned public split: **500 tasks** and
**1,952 normalized claims**. Reference trajectories are intentionally excluded.

## Datasets

| Dataset | Tasks | Intended use |
|---|---:|---|
| `mcp-atlas-smoke@1.0.5` | 1 | Credential-free Docker deployment check. |
| `mcp-atlas-credential-free@1.0.5` | 30 | Evaluation without Atlas service credentials. |
| `mcp-atlas@1.0.5` | 500 | Full public split; credentials and prepared external state may be required. |

Start with the smoke task:

```bash
harbor run \
  --repo Osmosis-AI/mcp-atlas-harbor@v1.0.5 \
  --dataset mcp-atlas-smoke@1.0.5 \
  --agent <agent> \
  --model <model>
```

The claim grader uses an OpenAI-compatible endpoint:

```bash
export EVAL_LLM_API_KEY='...'
export EVAL_LLM_BASE_URL='https://your-endpoint'   # a trailing /v1 is accepted
export EVAL_LLM_MODEL='your-judge-model'
```

The verifier scores an empty or `ERROR:` response as `0`. Verifier-side
failures (missing judge credentials, judge outages, unreadable claims) exit
non-zero without writing `reward.json`, so Harbor records a verifier error and
retries instead of reporting a zero score. The graded answer is an explicit
final-answer tool argument in the ATIF trajectory, otherwise the agent's last
message, otherwise `response.txt`, `final_answer.txt`, or `answer.txt` under
`/logs/agent`.

## Runtime

Each task uses the digest-pinned official MCP-Atlas image. Because that image
exposes Atlas REST endpoints rather than an MCP endpoint, the task includes a
small streamable-HTTP bridge:

```text
Harbor agent -> allowlisting MCP bridge -> official MCP-Atlas runtime
```

The bridge exposes only the task's enabled tools. Atlas credentials are
substituted only into the runtime sidecar; the bridge and main agent do not
receive them.

Local Docker is the supported environment. Keep concurrency low because every
trial starts an Atlas runtime and cold startup can take more than a minute.

### Credentialed tasks

465 tasks need at least one Atlas credential; 278 of them use a stateful
server (Airtable, Google Workspace, MongoDB, Notion, Slack) whose account must
first be seeded with the upstream `data_exports`. For those tasks, export the
required Atlas credentials in the host shell before starting Harbor:

```bash
export GITHUB_TOKEN='short-lived-token-for-a-disposable-account'
export BRAVE_API_KEY='...'
```

Do not put Atlas credentials in `--agent-env` or the Harbor agent
configuration. Some tasks also rely on external accounts prepared for the
original benchmark, so a valid key alone may not recreate their state.

## Evaluation fidelity

The adapter preserves prompts, available official tools, and MCP-Atlas's
per-claim grades: fulfilled = 1, partially fulfilled = 0.5, and not fulfilled =
0. The task reward is their arithmetic mean, rounded to three decimals.

The source `TRAJECTORY` column is discarded. Claims and oracle answers live in
Harbor verifier/solution surfaces and are not exposed to a normal agent.

Four distractor prefixes in the public data are absent from the pinned image:
`anili`, `balldontlie`, `f1-mcp-server`, and `rijksmuseum-server`. The adapter
removes those unavailable tools and records them in task metadata. They occur
in 85 tasks and are not called by the public reference trajectories.

Results from live external services can drift. Quantitative
original-versus-Harbor model parity has not yet been run; conversion and smoke
validation are not reported as model parity.

## Pinned provenance

| Input | Pin |
|---|---|
| Hugging Face dataset | `ScaleAI/MCP-Atlas` at `8c563b55d7c967755f474299848049834d624617` |
| Public Parquet | SHA-256 `2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0` |
| Upstream source | `scaleapi/mcp-atlas` at `f24ba3fb0bfa484c86acb28431fad6d7282455f9` |
| Harbor adapter | `Osmosis-AI/harbor` at `eba22020099645173401fc54c712f00e21537f76` |
| Official Atlas image | `ghcr.io/scaleapi/mcp-atlas:1.2.7@sha256:24e6ed3534916afe2c6825382da159a30e23516ef612be5d074fd96a74f9184c` |

`manifests/mcp-atlas-1.0.5.json` records these inputs and every generated task
checksum. The immutable `v1.0.5` tag pins this repository snapshot.

## Repository layout

```text
.
├── tasks/                          # 500 Harbor tasks
├── manifests/mcp-atlas-1.0.5.json # source pins and task checksums
├── registry.json                   # full, credential-free, and smoke views
└── scripts/release.sh              # regenerate, validate, and compare
```

No source exports, credentials, trajectories, image archives, or evaluation
outputs belong in this repository.

## Reproduce the release

The check script fetches the pinned Harbor adapter and public Parquet,
regenerates all tasks, validates them, and compares the result byte-for-byte:

```bash
./scripts/release.sh --check
```

You can use already verified local inputs:

```bash
HARBOR_SRC=/path/to/harbor \
MCP_ATLAS_SOURCE_FILE=/path/to/MCP-Atlas.parquet \
./scripts/release.sh --check
```

Published version tags are immutable and must not be moved.

## Licensing and attribution

Benchmark-derived prompts, claims, answers, and task data are CC BY 4.0; see
[LICENSE-DATA](LICENSE-DATA). Harbor-authored code and documentation are
Apache-2.0; see [LICENSE-CODE](LICENSE-CODE).

See [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[CITATION.bib](CITATION.bib) for attribution and citation details.
