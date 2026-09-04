#!/usr/bin/env python3
"""Score an MCP-Atlas response with the upstream per-claim rubric."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AGENT_DIR = Path("/logs/agent")
CLAIMS_PATH = Path("/tests/claims.json")
REWARD_PATH = Path("/logs/verifier/reward.json")
DETAILS_PATH = Path("/logs/verifier/coverage_details.json")
FALLBACK_FILENAMES = ("response.txt", "final_answer.txt", "answer.txt")
TERMINAL_TOOL_NAMES = {"final_answer", "mark_task_complete", "submit_answer"}
ANSWER_ARGUMENT_KEYS = ("final_answer", "answer", "response", "result", "text")
OUTCOME_SCORES = {
    "fulfilled": 1.0,
    "partially_fulfilled": 0.5,
    "not_fulfilled": 0.0,
}
CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_text": {"type": "string"},
        "coverage_outcome": {"type": "string", "enum": list(OUTCOME_SCORES)},
        "justification": {"type": "string"},
        "confidence_level": {"type": "number"},
    },
    "required": [
        "claim_text",
        "coverage_outcome",
        "justification",
        "confidence_level",
    ],
    "additionalProperties": False,
}


class VerifierError(RuntimeError):
    pass


class JudgeError(VerifierError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ExtractionResult(NamedTuple):
    response: str
    source: str


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    outcome: str
    justification: str
    confidence: float

    @property
    def score(self) -> float:
        return OUTCOME_SCORES[self.outcome]


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    concurrency: int
    max_attempts: int
    retry_base_sec: float
    timeout_sec: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> JudgeConfig:
        values = os.environ if env is None else env
        api_key = values.get("EVAL_LLM_API_KEY", "").strip()
        base_url = values.get("EVAL_LLM_BASE_URL", "").strip().rstrip("/")
        if not api_key:
            raise VerifierError("EVAL_LLM_API_KEY is required")
        if not base_url.startswith(("http://", "https://")):
            raise VerifierError("EVAL_LLM_BASE_URL must be an http(s) URL")
        try:
            config = cls(
                base_url,
                api_key,
                values.get("EVAL_LLM_MODEL", "gemini/gemini-3.1-pro-preview").strip(),
                int(values.get("EVAL_LLM_CONCURRENCY", "30")),
                int(values.get("EVAL_LLM_MAX_ATTEMPTS", "3")),
                float(values.get("EVAL_LLM_RETRY_BASE_SEC", "1")),
                float(values.get("EVAL_LLM_TIMEOUT_SEC", "60")),
            )
        except ValueError as exc:
            raise VerifierError("judge numeric configuration is invalid") from exc
        if min(config.concurrency, config.max_attempts, config.timeout_sec) <= 0:
            raise VerifierError(
                "judge concurrency, attempts, and timeout must be positive"
            )
        if config.retry_base_sec < 0:
            raise VerifierError("judge retry delay cannot be negative")
        return config


def _message_text(message: object) -> str:
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, list):
        return ""
    texts = []
    for part in message:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def _terminal_answer(step: Mapping[str, Any]) -> str:
    calls = step.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    for call in reversed(calls):
        if not isinstance(call, Mapping):
            continue
        function_name = call.get("function_name")
        if not isinstance(function_name, str):
            continue
        name = function_name.lower().replace("-", "_").rsplit(".", 1)[-1]
        arguments = call.get("arguments")
        if name not in TERMINAL_TOOL_NAMES or not isinstance(arguments, dict):
            continue
        for key in ANSWER_ARGUMENT_KEYS:
            answer = arguments.get(key)
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    return ""


def extract_response(agent_dir: Path = AGENT_DIR) -> ExtractionResult:
    """Prefer the last main-agent ATIF message, then explicit text files."""

    trajectory_path = agent_dir / "trajectory.json"
    if trajectory_path.is_file() and not trajectory_path.is_symlink():
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            steps = trajectory["steps"]
            if not isinstance(steps, list):
                raise TypeError("steps is not a list")
            for step in reversed(steps):
                if (
                    not isinstance(step, Mapping)
                    or step.get("source") != "agent"
                    or step.get("is_copied_context") is True
                ):
                    continue
                answer = _terminal_answer(cast(Mapping[str, Any], step))
                if answer:
                    return ExtractionResult(answer, "atif_terminal_tool")
                answer = _message_text(step.get("message"))
                if answer:
                    return ExtractionResult(answer, "atif")
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            pass

    for filename in FALLBACK_FILENAMES:
        path = agent_dir / filename
        if path.is_symlink():
            continue
        try:
            answer = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if answer:
            return ExtractionResult(answer, filename)
    return ExtractionResult("", "none")


def load_claims(path: Path = CLAIMS_PATH) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerifierError(f"cannot read claims.json: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise VerifierError("claims.json must be a non-empty list")
    if any(not isinstance(claim, str) or not claim.strip() for claim in raw):
        raise VerifierError("every claim must be a non-empty string")
    return [claim.strip() for claim in raw]


def _judge_payload(model: str, claim: str, response: str) -> dict[str, Any]:
    prompt = f"""Evaluate whether the response addresses the claim.
Use one coverage_outcome: fulfilled (all key details), partially_fulfilled
(some key details), or not_fulfilled (not substantively addressed).
Allow reasonable numerical rounding. Return a short justification and a
confidence_level from 0 to 1.

CLAIM TO EVALUATE:
{claim}

MODEL RESPONSE TO ANALYZE:
{response}"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claim_evaluation", "schema": CLAIM_SCHEMA},
        },
    }


def _http_post_json(
    url: str, payload: Mapping[str, Any], api_key: str, timeout: float
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode())
    except HTTPError as exc:
        retryable = exc.code == 429 or exc.code >= 500
        raise JudgeError(
            f"judge returned HTTP {exc.code}", retryable=retryable
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise JudgeError(f"judge request failed: {exc}", retryable=True) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise JudgeError(f"judge returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeError("judge response must be a JSON object")
    return data


def _parse_evaluation(data: Mapping[str, Any], claim: str) -> ClaimEvaluation:
    try:
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise JudgeError("judge response has invalid structured content") from exc
    if not isinstance(result, dict):
        raise JudgeError("judge content must be a JSON object")
    outcome = result.get("coverage_outcome")
    justification = result.get("justification")
    confidence = result.get("confidence_level")
    if outcome not in OUTCOME_SCORES:
        raise JudgeError(f"unknown coverage_outcome: {outcome!r}")
    if not isinstance(justification, str) or not justification.strip():
        raise JudgeError("judge result has no justification")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise JudgeError("judge confidence_level must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise JudgeError("judge confidence_level must be between 0 and 1")
    return ClaimEvaluation(claim, outcome, justification.strip(), float(confidence))


Transport = Callable[[str, Mapping[str, Any], str, float], dict[str, Any]]


async def evaluate_claims(
    claims: Sequence[str],
    response: str,
    config: JudgeConfig,
    transport: Transport = _http_post_json,
) -> list[ClaimEvaluation]:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def evaluate(claim: str) -> ClaimEvaluation:
        async with semaphore:
            for attempt in range(config.max_attempts):
                try:
                    data = await asyncio.to_thread(
                        transport,
                        f"{config.base_url}/v1/chat/completions",
                        _judge_payload(config.model, claim, response),
                        config.api_key,
                        config.timeout_sec,
                    )
                    return _parse_evaluation(data, claim)
                except JudgeError as exc:
                    if not exc.retryable or attempt + 1 == config.max_attempts:
                        raise
                    await asyncio.sleep(config.retry_base_sec * (2**attempt))
            raise JudgeError("judge request failed")

    tasks: list[asyncio.Task[ClaimEvaluation]] = []
    async with asyncio.TaskGroup() as group:
        for claim in claims:
            tasks.append(group.create_task(evaluate(claim)))
    return [task.result() for task in tasks]


def aggregate_results(results: Sequence[ClaimEvaluation]) -> dict[str, Any]:
    coverage = round(sum(result.score for result in results) / len(results), 3)
    return {
        "coverage_score": coverage,
        "total_claims": len(results),
        "per_claim": [
            {
                "claim": result.claim,
                "coverage_outcome": result.outcome,
                "score": result.score,
                "justification": result.justification,
                "confidence_level": result.confidence,
            }
            for result in results
        ],
    }


def _write_outputs(
    aggregate: Mapping[str, Any],
    source: str,
    error: str | None,
    reward_path: Path,
    details_path: Path,
) -> None:
    coverage = float(aggregate["coverage_score"])
    reward = {
        "reward": coverage,
        "coverage_score": coverage,
        "pass_at_0_50": float(coverage >= 0.50),
        "pass_at_0_75": float(coverage >= 0.75),
    }
    details = {**aggregate, "response_source": source, "error": error}
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(reward, indent=2) + "\n", encoding="utf-8")
    details_path.write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


async def run_verifier(
    *,
    agent_dir: Path = AGENT_DIR,
    claims_path: Path = CLAIMS_PATH,
    reward_path: Path = REWARD_PATH,
    details_path: Path = DETAILS_PATH,
    env: Mapping[str, str] | None = None,
    transport: Transport = _http_post_json,
) -> dict[str, Any]:
    claims: list[str] = []
    extraction = ExtractionResult("", "none")
    try:
        claims = load_claims(claims_path)
        extraction = extract_response(agent_dir)
        if not extraction.response or extraction.response.startswith("ERROR:"):
            raise VerifierError("model response is empty or reports an error")
        config = JudgeConfig.from_env(env)
        aggregate = aggregate_results(
            await evaluate_claims(claims, extraction.response, config, transport)
        )
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failed = [
            ClaimEvaluation(claim, "not_fulfilled", error, 1.0) for claim in claims
        ]
        aggregate = (
            aggregate_results(failed)
            if failed
            else {
                "coverage_score": 0.0,
                "total_claims": 0,
                "per_claim": [],
            }
        )
    _write_outputs(aggregate, extraction.source, error, reward_path, details_path)
    return {**aggregate, "error": error}


def main() -> int:
    result = asyncio.run(run_verifier())
    if result["error"]:
        print(f"MCP-Atlas verifier scored zero: {result['error']}", file=sys.stderr)
    else:
        print(f"MCP-Atlas coverage: {result['coverage_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
