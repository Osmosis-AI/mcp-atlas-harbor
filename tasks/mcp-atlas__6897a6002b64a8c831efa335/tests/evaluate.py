#!/usr/bin/env python3
"""Score an MCP-Atlas answer with the upstream per-claim coverage rubric.

The verifier intentionally uses only the Python 3.12 standard library so it can
run in arbitrary benchmark images.  It reads the main ATIF trajectory (including
safe continuation files), ignores copied context and embedded subagent traces,
and falls back only to a small set of explicit final-answer files.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AGENT_DIR = Path("/logs/agent")
CLAIMS_PATH = Path("/tests/claims.json")
REWARD_PATH = Path("/logs/verifier/reward.json")
DETAILS_PATH = Path("/logs/verifier/coverage_details.json")

MAX_RESPONSE_CHARS = 500_000
MAX_JUDGE_RESPONSE_BYTES = 10 * 1024 * 1024
TRUNCATION_MARKER = "\n\n[TRUNCATED — original response was too long]"
FALLBACK_FILENAMES = ("response.txt", "final_answer.txt", "answer.txt")
MINI_SWE_COMPLETION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
OUTCOME_SCORES = {
    "fulfilled": 1.0,
    "partially_fulfilled": 0.5,
    "not_fulfilled": 0.0,
}

# These are orchestration-level completion calls, not MCP server tools.  A
# generic action call is also terminal when its arguments explicitly say so.
TERMINAL_TOOL_NAMES = frozenset(
    {
        "answer",
        "attempt_completion",
        "complete",
        "done",
        "final_answer",
        "finish",
        "mark_done",
        "mark_task_complete",
        "respond",
        "submit",
        "submit_answer",
        "submit_and_exit",
        "task_done",
        "terminate",
    }
)
TERMINAL_ACTION_TYPES = frozenset(
    {"answer", "complete", "done", "finish", "final", "terminate"}
)
ANSWER_ARGUMENT_KEYS = (
    "final_answer",
    "answer",
    "response",
    "result",
    "summary",
    "message",
    "text",
    "output",
)
ANSWER_CONTAINER_KEYS = ("payload", "data", "content", "value")

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_text": {"type": "string"},
        "coverage_outcome": {
            "type": "string",
            "enum": list(OUTCOME_SCORES),
        },
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
    """Base class for actionable verifier failures."""


class ConfigurationError(VerifierError):
    """Raised before judging when required credentials are absent or invalid."""


class TrajectoryError(VerifierError):
    """Raised when an ATIF trajectory or continuation is unsafe or malformed."""


class JudgeError(VerifierError):
    """Base class for judge request and response failures."""


class AuthenticationError(JudgeError):
    """A 401/403 judge response, which must never be retried."""


class RetryableJudgeError(JudgeError):
    """A transient judge failure that may be retried a bounded number of times."""


class NonRetryableJudgeError(JudgeError):
    """A permanent judge failure other than invalid credentials."""


@dataclass(frozen=True)
class ExtractionResult:
    response: str
    source: str
    note: str | None = None
    authoritative: bool = False


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_keys: tuple[str, ...]
    model: str
    concurrency: int = 30
    max_attempts: int = 3
    retry_base_sec: float = 1.0
    timeout_sec: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> JudgeConfig:
        values = os.environ if env is None else env
        raw_keys = values.get("EVAL_LLM_API_KEY") or values.get("LLM_API_KEY", "")
        api_keys = tuple(key.strip() for key in raw_keys.split(",") if key.strip())
        if not api_keys:
            raise ConfigurationError(
                "Judge API key not found; set EVAL_LLM_API_KEY (or LLM_API_KEY)."
            )

        base_url = (
            (values.get("EVAL_LLM_BASE_URL") or values.get("LLM_BASE_URL", ""))
            .strip()
            .rstrip("/")
        )
        if not base_url:
            raise ConfigurationError(
                "Judge base URL not found; set EVAL_LLM_BASE_URL (or LLM_BASE_URL)."
            )
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                "Judge base URL must start with http:// or https://."
            )

        concurrency_env = (
            "EVAL_LLM_CONCURRENCY"
            if values.get("EVAL_LLM_CONCURRENCY", "").strip()
            else "MCP_ATLAS_JUDGE_CONCURRENCY"
        )
        return cls(
            base_url=base_url,
            api_keys=api_keys,
            model=(
                values.get("EVAL_LLM_MODEL")
                or values.get("JUDGE_MODEL")
                or "gemini/gemini-3.1-pro-preview"
            ).strip(),
            concurrency=_bounded_int(values, concurrency_env, 30, 1, 100),
            max_attempts=_bounded_int(values, "EVAL_LLM_MAX_ATTEMPTS", 3, 1, 10),
            retry_base_sec=_bounded_float(
                values, "EVAL_LLM_RETRY_BASE_SEC", 1.0, 0.0, 60.0
            ),
            timeout_sec=_bounded_float(
                values, "EVAL_LLM_TIMEOUT_SEC", 60.0, 1.0, 600.0
            ),
        )


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    coverage_outcome: str
    justification: str
    confidence_level: float
    judge_error: bool = False

    @property
    def score(self) -> float:
        return OUTCOME_SCORES[self.coverage_outcome]


def _bounded_int(
    env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _bounded_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrajectoryError(
            f"Cannot read ATIF trajectory {path.name}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TrajectoryError(
            f"Invalid JSON in ATIF trajectory {path.name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TrajectoryError(
            f"ATIF trajectory {path.name} must contain a JSON object."
        )
    return value


def _safe_continuation_path(agent_dir: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise TrajectoryError("continued_trajectory_ref must be a non-empty string.")

    base = agent_dir.resolve()
    candidate = agent_dir / reference
    if candidate.is_symlink():
        raise TrajectoryError(
            f"Unsafe ATIF continuation {reference!r}; symlinks are not allowed."
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TrajectoryError(
            f"ATIF continuation {reference!r} does not resolve to a readable file."
        ) from exc
    if resolved.parent != base or not resolved.is_file():
        raise TrajectoryError(
            f"Unsafe ATIF continuation {reference!r}; it must be a file in {base}."
        )
    return resolved


def load_main_atif_steps(agent_dir: Path = AGENT_DIR) -> list[dict[str, Any]]:
    """Load only the main ATIF continuation chain in chronological order."""

    current = agent_dir / "trajectory.json"
    if current.is_symlink():
        raise TrajectoryError("Unsafe ATIF trajectory.json; symlinks are not allowed.")
    if not current.is_file():
        return []
    base = agent_dir.resolve()
    try:
        resolved_initial = current.resolve(strict=True)
    except OSError as exc:
        raise TrajectoryError("ATIF trajectory.json is not a readable file.") from exc
    if resolved_initial.parent != base or not resolved_initial.is_file():
        raise TrajectoryError(
            "Unsafe ATIF trajectory.json outside the agent directory."
        )
    current = resolved_initial

    all_steps: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for _ in range(128):
        resolved = current.resolve()
        if resolved in seen:
            raise TrajectoryError(
                f"Cycle in ATIF continuation chain at {current.name}."
            )
        seen.add(resolved)

        trajectory = _read_json_object(current)
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            raise TrajectoryError(f"ATIF trajectory {current.name} has no steps array.")
        all_steps.extend(step for step in steps if isinstance(step, dict))

        continued_ref = trajectory.get("continued_trajectory_ref")
        if continued_ref is None:
            return all_steps
        current = _safe_continuation_path(agent_dir, continued_ref)

    raise TrajectoryError("ATIF continuation chain exceeds 128 files.")


def _message_text(message: object) -> str:
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for part in message:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def _normalized_tool_name(call: Mapping[str, Any]) -> str:
    raw_name = call.get("function_name") or call.get("name")
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.strip().lower().replace("-", "_")
    for separator in (".", "/", ":"):
        name = name.rsplit(separator, 1)[-1]
    return name


def _call_arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = call.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return decoded
    return {}


def _is_terminal_call(call: Mapping[str, Any]) -> bool:
    if _normalized_tool_name(call) in TERMINAL_TOOL_NAMES:
        return True
    arguments = _call_arguments(call)
    for key in ("type", "action", "status"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip().lower() in TERMINAL_ACTION_TYPES:
            return True
    return False


def _is_mini_swe_completion_call(call: Mapping[str, Any]) -> bool:
    """Recognize mini-swe-agent's exact final-output submission sentinel."""

    arguments = call.get("arguments")
    return (
        call.get("function_name") == "bash"
        and isinstance(arguments, Mapping)
        and arguments.get("command") == MINI_SWE_COMPLETION_COMMAND
    )


def _answer_from_value(value: object, depth: int = 0) -> str:
    if isinstance(value, str):
        return value.strip()
    if depth >= 2 or not isinstance(value, dict):
        return ""
    for key in ANSWER_ARGUMENT_KEYS:
        answer = _answer_from_value(value.get(key), depth + 1)
        if answer:
            return answer
    for key in ANSWER_CONTAINER_KEYS:
        answer = _answer_from_value(value.get(key), depth + 1)
        if answer:
            return answer
    return ""


def _terminal_answer(tool_calls: Sequence[Mapping[str, Any]]) -> str:
    """Extract an answer from a terminal marker and its immediately prior action."""

    for call in reversed(tool_calls):
        if not _is_terminal_call(call):
            continue
        answer = _answer_from_value(_call_arguments(call))
        if answer:
            return answer
    return ""


def _producer_completion(
    step: Mapping[str, Any], producer: str
) -> ExtractionResult | None:
    """Read a producer-owned completion state without guessing from its message."""

    extra = step.get("extra")
    if not isinstance(extra, Mapping):
        return None
    producer_extra = extra.get(producer)
    if not isinstance(producer_extra, Mapping):
        return None
    label = "terminus" if producer == "terminus_2" else producer
    note_label = label
    if producer == "terminus_2" and producer_extra.get("raw_content") is True:
        note_label = "terminus_raw"
    completion = producer_extra.get("task_completion")
    if not isinstance(completion, Mapping):
        return ExtractionResult(
            "", "atif", f"invalid_{note_label}_completion", authoritative=True
        )
    status = completion.get("status")
    if status != "confirmed":
        return ExtractionResult(
            "", "atif", f"{note_label}_{status or 'unknown'}", authoritative=True
        )
    answer = completion.get("final_answer")
    if isinstance(answer, str) and answer.strip():
        return ExtractionResult(
            answer.strip(), f"atif_{label}_completion", authoritative=True
        )
    return ExtractionResult(
        "",
        "atif",
        f"{note_label}_completion_without_answer",
        authoritative=True,
    )


def extract_atif_response(agent_dir: Path = AGENT_DIR) -> ExtractionResult:
    """Extract the committed final response from the main ATIF trajectory."""

    steps = load_main_atif_steps(agent_dir)
    for step in reversed(steps):
        if step.get("is_copied_context") is True:
            continue

        extra = step.get("extra")
        if isinstance(extra, Mapping) and extra.get("is_sidechain") is True:
            continue

        source = step.get("source")
        if source == "user":
            return ExtractionResult(
                "",
                "atif",
                "no_agent_response_after_last_user",
                authoritative=True,
            )
        if source != "agent":
            continue

        for producer in ("terminus_2", "computer_1"):
            completion = _producer_completion(step, producer)
            if completion is not None:
                return completion

        text = _message_text(step.get("message"))
        raw_calls = step.get("tool_calls")
        tool_calls = (
            [call for call in raw_calls if isinstance(call, dict)]
            if isinstance(raw_calls, list)
            else []
        )
        has_observation = step.get("observation") not in (None, "", [], {})
        has_reasoning = bool(_message_text(step.get("reasoning_content")))
        has_inference_metadata = (
            step.get("metrics") not in (None, "", [], {})
            or isinstance(step.get("llm_call_count"), int)
            and step.get("llm_call_count", 0) > 0
        )

        if (
            isinstance(extra, Mapping)
            and extra.get("source_call_id")
            and not tool_calls
        ):
            return ExtractionResult("", "atif", "trailing_orphan_tool_result")

        # Empty bookkeeping records do not hide an earlier final response.
        if (
            not text
            and not tool_calls
            and not has_observation
            and not has_reasoning
            and not has_inference_metadata
        ):
            continue

        if not text and not tool_calls and (has_reasoning or has_inference_metadata):
            return ExtractionResult("", "atif", "trailing_reasoning_without_answer")

        if tool_calls:
            # Harbor appends mark_task_complete after the action it commits.  If
            # the final call is anything else, the agent stopped mid-tool-loop.
            # mini-swe-agent instead uses an exact bash sentinel and stores the
            # submitted final output in the same ATIF step's message.
            if _is_mini_swe_completion_call(tool_calls[-1]):
                if text:
                    return ExtractionResult(text, "atif_mini_swe_completion")
                return ExtractionResult("", "atif", "terminal_call_without_answer")
            if not _is_terminal_call(tool_calls[-1]):
                return ExtractionResult("", "atif", "trailing_nonterminal_tool_call")
            answer = _terminal_answer(tool_calls)
            if answer:
                return ExtractionResult(answer, "atif_terminal_tool")
            return ExtractionResult("", "atif", "terminal_call_without_answer")

        if text:
            return ExtractionResult(text, "atif")
        return ExtractionResult("", "atif", "trailing_observation_without_answer")

    return ExtractionResult("", "atif", "no_agent_response")


def _fallback_response(agent_dir: Path) -> ExtractionResult | None:
    base = agent_dir.resolve()
    for filename in FALLBACK_FILENAMES:
        path = agent_dir / filename
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve(strict=True)
            if resolved.parent != base or not resolved.is_file():
                continue
            text = resolved.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return ExtractionResult(text, filename)
    return None


def extract_response(agent_dir: Path = AGENT_DIR) -> ExtractionResult:
    """Prefer main ATIF, then only explicit final-answer filename contracts."""

    atif_note: str | None = None
    try:
        atif = extract_atif_response(agent_dir)
        if atif.response:
            return atif
        if atif.authoritative:
            return atif
        atif_note = atif.note
    except TrajectoryError as exc:
        atif_note = str(exc)

    fallback = _fallback_response(agent_dir)
    if fallback is not None:
        return fallback
    return ExtractionResult("", "none", atif_note or "no response artifact found")


def load_claims(path: Path = CLAIMS_PATH) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerifierError(f"Cannot read claims file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerifierError(f"Invalid JSON in claims file {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("claims")
    if not isinstance(raw, list):
        raise VerifierError("claims.json must contain a JSON list of claim strings.")
    claims = [
        claim.strip() for claim in raw if isinstance(claim, str) and claim.strip()
    ]
    if len(claims) != len(raw):
        raise VerifierError("Every entry in claims.json must be a non-empty string.")
    return claims


def truncate_response(response: str) -> tuple[str, bool]:
    if len(response) <= MAX_RESPONSE_CHARS:
        return response, False
    return response[:MAX_RESPONSE_CHARS] + TRUNCATION_MARKER, True


def build_claim_prompt(claim: str, response: str) -> str:
    return f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."""


def _judge_payload(model: str, prompt: str) -> dict[str, Any]:
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
    url: str, payload: Mapping[str, Any], api_key: str, timeout_sec: float
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            raw = response.read(MAX_JUDGE_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        error_body = exc.read(2_048).decode("utf-8", errors="replace")
        message = f"Judge API returned HTTP {exc.code}: {error_body}"
        if exc.code in {401, 403}:
            raise AuthenticationError(message) from exc
        if exc.code in {408, 409, 425, 429, 500, 502, 503, 504}:
            raise RetryableJudgeError(message) from exc
        raise NonRetryableJudgeError(message) from exc
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise RetryableJudgeError(f"Judge request failed: {exc}") from exc

    if len(raw) > MAX_JUDGE_RESPONSE_BYTES:
        raise RetryableJudgeError("Judge response exceeded 10 MiB.")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetryableJudgeError(f"Judge returned invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RetryableJudgeError("Judge response must be a JSON object.")
    return decoded


def _message_content(data: Mapping[str, Any]) -> object:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RetryableJudgeError("Judge response has no choices[0].message.") from exc
    if not isinstance(message, dict):
        raise RetryableJudgeError("Judge response message is not an object.")
    if isinstance(message.get("parsed"), dict):
        return message["parsed"]
    return message.get("content")


def _parse_structured_evaluation(
    data: Mapping[str, Any], claim: str
) -> ClaimEvaluation:
    content = _message_content(data)
    if isinstance(content, dict):
        result = content
    else:
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            content = "".join(text_parts)
        if not isinstance(content, str):
            raise RetryableJudgeError("Judge message content is not structured JSON.")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RetryableJudgeError(
                f"Judge message content is invalid structured JSON: {exc}"
            ) from exc

    if not isinstance(result, dict):
        raise RetryableJudgeError("Structured judge result is not an object.")
    outcome = result.get("coverage_outcome")
    justification = result.get("justification")
    confidence = result.get("confidence_level")
    if not isinstance(outcome, str) or outcome not in OUTCOME_SCORES:
        raise RetryableJudgeError(f"Unknown coverage_outcome: {outcome!r}.")
    if not isinstance(justification, str) or not justification.strip():
        raise RetryableJudgeError("Judge result has no justification.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RetryableJudgeError("Judge confidence_level is not numeric.")
    numeric_confidence = float(confidence)
    if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
        raise RetryableJudgeError(
            "Judge confidence_level must be finite and in [0, 1]."
        )
    return ClaimEvaluation(
        claim=claim,
        coverage_outcome=outcome,
        justification=justification.strip(),
        confidence_level=numeric_confidence,
    )


Transport = Callable[[str, Mapping[str, Any], str, float], dict[str, Any]]
Sleep = Callable[[float], Awaitable[None]]


class JudgeClient:
    def __init__(
        self,
        config: JudgeConfig,
        *,
        transport: Transport = _http_post_json,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.config = config
        self.transport = transport
        self.sleep = sleep
        self.url = f"{config.base_url}/v1/chat/completions"

    async def evaluate_claim(
        self, claim: str, response: str, request_index: int = 0
    ) -> ClaimEvaluation:
        payload = _judge_payload(self.config.model, build_claim_prompt(claim, response))
        last_error: JudgeError | None = None
        for attempt in range(self.config.max_attempts):
            api_key = self.config.api_keys[
                (request_index + attempt) % len(self.config.api_keys)
            ]
            try:
                data = await asyncio.to_thread(
                    self.transport,
                    self.url,
                    payload,
                    api_key,
                    self.config.timeout_sec,
                )
                return _parse_structured_evaluation(data, claim)
            except AuthenticationError:
                raise
            except NonRetryableJudgeError:
                raise
            except RetryableJudgeError as exc:
                last_error = exc
            except (TimeoutError, OSError) as exc:
                last_error = RetryableJudgeError(f"Judge request failed: {exc}")

            if attempt + 1 < self.config.max_attempts:
                await self.sleep(self.config.retry_base_sec * (2**attempt))

        if last_error is None:
            raise RetryableJudgeError("Judge request failed without an error message.")
        raise last_error


def _failed_evaluation(claim: str, reason: str) -> ClaimEvaluation:
    return ClaimEvaluation(
        claim=claim,
        coverage_outcome="not_fulfilled",
        justification=f"Evaluation failed: {reason}",
        confidence_level=0.1,
        judge_error=True,
    )


async def evaluate_claims(
    claims: Sequence[str],
    response: str,
    client: JudgeClient,
    *,
    concurrency: int | None = None,
) -> list[ClaimEvaluation]:
    """Evaluate claims concurrently while preserving input order."""

    limit = client.config.concurrency if concurrency is None else concurrency
    if limit < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(limit)
    authentication_failed = asyncio.Event()
    results: list[ClaimEvaluation | None] = [None] * len(claims)

    async def evaluate_one(index: int, claim: str) -> None:
        async with semaphore:
            if authentication_failed.is_set():
                return
            try:
                results[index] = await client.evaluate_claim(claim, response, index)
            except AuthenticationError:
                authentication_failed.set()
                raise
            except JudgeError as exc:
                results[index] = _failed_evaluation(claim, str(exc))

    authentication_errors: list[AuthenticationError] = []
    try:
        async with asyncio.TaskGroup() as task_group:
            for index, claim in enumerate(claims):
                task_group.create_task(evaluate_one(index, claim))
    except* AuthenticationError as exc_group:
        authentication_errors.extend(
            exc for exc in exc_group.exceptions if isinstance(exc, AuthenticationError)
        )

    if authentication_errors:
        raise authentication_errors[0]
    if any(result is None for result in results):
        raise JudgeError("Claim evaluation did not produce a result for every claim.")
    return [result for result in results if result is not None]


def zero_evaluations(claims: Sequence[str], reason: str) -> list[ClaimEvaluation]:
    return [
        ClaimEvaluation(
            claim=claim,
            coverage_outcome="not_fulfilled",
            justification=reason,
            confidence_level=1.0,
        )
        for claim in claims
    ]


def aggregate_results(results: Sequence[ClaimEvaluation]) -> dict[str, Any]:
    total = len(results)
    coverage = (
        round(sum(result.score for result in results) / total, 3) if total else 0.0
    )
    confidence = (
        sum(result.confidence_level for result in results) / total if total else 1.0
    )
    return {
        "coverage_score": coverage,
        "total_claims": total,
        "fully_covered_claims": sum(result.score == 1.0 for result in results),
        "partially_covered_claims": sum(result.score == 0.5 for result in results),
        "evaluation_confidence": confidence,
        "judge_errors": sum(result.judge_error for result in results),
        "per_claim": [
            {
                "claim": result.claim,
                "coverage_outcome": result.coverage_outcome,
                "score": result.score,
                "covered": (
                    True
                    if result.score == 1.0
                    else "partial"
                    if result.score == 0.5
                    else False
                ),
                "reason": result.justification,
                "confidence_level": result.confidence_level,
                "judge_error": result.judge_error,
            }
            for result in results
        ],
    }


def reward_metrics(coverage: float) -> dict[str, float]:
    if not math.isfinite(coverage):
        raise VerifierError("Coverage score must be finite.")
    return {
        "reward": coverage,
        "coverage_score": coverage,
        "pass_at_0_50": float(coverage >= 0.50),
        "pass_at_0_75": float(coverage >= 0.75),
    }


def write_outputs(
    aggregate: Mapping[str, Any],
    extraction: ExtractionResult,
    *,
    original_response_chars: int,
    truncated: bool,
    judge_model: str | None,
    reward_path: Path = REWARD_PATH,
    details_path: Path = DETAILS_PATH,
) -> None:
    coverage = float(aggregate["coverage_score"])

    # reward.json must remain a flat object whose values are finite numbers.
    rewards = reward_metrics(coverage)
    details = {
        **aggregate,
        "response": {
            "source": extraction.source,
            "note": extraction.note,
            "original_chars": original_response_chars,
            "truncated": truncated,
        },
        "judge_model": judge_model,
    }
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(
        json.dumps(rewards, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    details_path.write_text(
        json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_empty_or_error_response(response: str) -> bool:
    stripped = response.strip()
    return not stripped or stripped.startswith("ERROR:")


async def run_verifier(
    *,
    agent_dir: Path = AGENT_DIR,
    claims_path: Path = CLAIMS_PATH,
    reward_path: Path = REWARD_PATH,
    details_path: Path = DETAILS_PATH,
    env: Mapping[str, str] | None = None,
    transport: Transport = _http_post_json,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    claims = load_claims(claims_path)
    extraction = extract_response(agent_dir)
    original_response_chars = len(extraction.response)
    response, truncated = truncate_response(extraction.response)

    if _is_empty_or_error_response(response):
        aggregate = aggregate_results(
            zero_evaluations(claims, "Empty or error response")
        )
        write_outputs(
            aggregate,
            extraction,
            original_response_chars=original_response_chars,
            truncated=truncated,
            judge_model=None,
            reward_path=reward_path,
            details_path=details_path,
        )
        return aggregate

    # Constructing configuration before scheduling any calls makes missing or
    # malformed credentials a deterministic fail-fast error.
    config = JudgeConfig.from_env(env)
    client = JudgeClient(config, transport=transport, sleep=sleep)
    results = await evaluate_claims(claims, response, client)
    aggregate = aggregate_results(results)
    write_outputs(
        aggregate,
        extraction,
        original_response_chars=original_response_chars,
        truncated=truncated,
        judge_model=config.model,
        reward_path=reward_path,
        details_path=details_path,
    )
    return aggregate


def main() -> int:
    # Never let a failed rerun leave a stale successful score behind.
    REWARD_PATH.unlink(missing_ok=True)
    DETAILS_PATH.unlink(missing_ok=True)
    try:
        aggregate = asyncio.run(run_verifier())
    except VerifierError as exc:
        print(f"MCP-Atlas verifier failed: {exc}", file=sys.stderr)
        return 1
    print(
        "MCP-Atlas coverage: "
        f"{aggregate['coverage_score']:.3f} across {aggregate['total_claims']} claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
