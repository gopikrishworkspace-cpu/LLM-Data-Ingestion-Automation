"""
llm_client.py — HTTP client for a local LLM (Ollama-compatible API).

Responsibilities:
  - Send prompts to a local LLM over HTTP (POST /api/generate)
  - Configurable model, base URL, timeout, retry count
  - Stack-based JSON extraction (handles nested braces correctly)
  - Prompt size guard to prevent accidental context overflow
  - Exponential backoff with jitter on transient failures
  - Streaming disabled by default (we need full JSON responses)

All intermediate prompt/response data stays in memory — nothing written
to disk.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Conservative default — most local models have 4k–8k context.
# Prompt + system together should stay well under model context window.
_DEFAULT_MAX_PROMPT_CHARS = 32_000


@dataclass
class LLMConfig:
    """Tunable parameters for the LLM client."""

    base_url: str = "http://localhost:11434"  # Ollama default
    model: str = "llama3"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_base_delay: float = 2.0     # seconds; doubled each retry
    retry_jitter: float = 0.5         # ± random jitter factor
    temperature: float = 0.1          # low temp for structured extraction
    max_tokens: int = 4096
    max_prompt_chars: int = _DEFAULT_MAX_PROMPT_CHARS
    # Ollama-specific options passed in the "options" payload key
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be > 0")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromptTooLargeError(ValueError):
    """Raised when prompt + system exceeds the configured size limit."""


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResponse:
    """Parsed response from the LLM."""

    raw_text: str               # full text returned by the model
    parsed_json: Any | None     # extracted JSON (dict / list), or None
    model: str
    duration_ms: int            # total round-trip time
    success: bool               # True if HTTP 200 and text is non-empty
    error: str | None           # human-readable error description, or None

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Stack-based JSON extraction
# ---------------------------------------------------------------------------

# Fenced code block (```json ... ```)
_JSON_FENCE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of JSON from potentially messy LLM output.

    Strategy (in order):
      1. Try full text as JSON.
      2. Look for ```json fenced blocks.
      3. Stack-based scan for the first balanced { } or [ ].
      4. Give up → return None.
    """
    # 1. Full text
    cleaned = text.strip()
    parsed = _try_parse(cleaned)
    if parsed is not None:
        return parsed

    # 2. Fenced block
    m = _JSON_FENCE.search(text)
    if m:
        parsed = _try_parse(m.group(1).strip())
        if parsed is not None:
            return parsed

    # 3. Stack-based balanced extraction
    candidate = _extract_balanced(text)
    if candidate is not None:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    logger.debug("JSON extraction failed for text: %.200s…", text)
    return None


def _extract_balanced(text: str) -> str | None:
    """Walk through text to find the first balanced JSON object or array.

    Uses a character-level stack that respects:
      - Nested { } and [ ]
      - String literals (skipping escaped quotes)

    Returns the balanced substring, or None if nothing found.
    """
    openers = {"{": "}", "[": "]"}
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in openers:
            i += 1
            continue

        # Found an opener — start stack walk
        start = i
        stack: list[str] = [openers[ch]]
        i += 1
        in_string = False

        while i < n and stack:
            c = text[i]

            if in_string:
                if c == "\\" and i + 1 < n:
                    i += 2  # skip escaped char
                    continue
                if c == '"':
                    in_string = False
                i += 1
                continue

            if c == '"':
                in_string = True
                i += 1
                continue

            if c in openers:
                stack.append(openers[c])
            elif c == stack[-1]:
                stack.pop()

            i += 1

        if not stack:
            return text[start:i]

        # Stack never balanced from this opener — try next occurrence
        # (don't reset i; we already walked past it)

    return None


def _try_parse(s: str) -> Any | None:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# HTTP transport (stdlib only — no external deps)
# ---------------------------------------------------------------------------


class LLMClient:
    """Synchronous HTTP client for a local Ollama-compatible LLM API.

    Usage::

        client = LLMClient(LLMConfig(model="llama3"))
        response = client.generate(
            prompt="Extract entities from: ...",
            system="You are a structured data extractor. Respond only in JSON.",
        )
        if response.success and response.parsed_json:
            entities = response.parsed_json
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._endpoint = f"{self.config.base_url.rstrip('/')}/api/generate"

    def generate(
        self,
        prompt: str,
        system: str,
        expect_json: bool = True,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a parsed response.

        Args:
            prompt: The user/instruction prompt.
            system: System prompt — REQUIRED. Defines the LLM's role
                    and output format expectations.
            expect_json: If True, attempt JSON extraction from the response.

        Returns:
            LLMResponse with raw text and (optionally) parsed JSON.

        Raises:
            PromptTooLargeError: if prompt + system exceeds max_prompt_chars.
        """
        # --- prompt size guard ---
        total_chars = len(prompt) + len(system)
        if total_chars > self.config.max_prompt_chars:
            msg = (
                f"Prompt too large: {total_chars} chars "
                f"(limit {self.config.max_prompt_chars})"
            )
            logger.error(msg)
            raise PromptTooLargeError(msg)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra_options,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        start_ms = _now_ms()

        raw_text = ""
        last_error: str | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_text = self._post(body)
                parsed = extract_json(raw_text) if expect_json else None
                duration = _now_ms() - start_ms

                if expect_json and parsed is None:
                    last_error = (
                        f"JSON extraction failed on attempt {attempt}"
                    )
                    logger.warning(
                        "Attempt %d/%d: LLM returned non-JSON response "
                        "(%.100s…), retrying",
                        attempt, self.config.max_retries, raw_text,
                    )
                    self._backoff(attempt)
                    continue

                return LLMResponse(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    model=self.config.model,
                    duration_ms=duration,
                    success=True,
                    error=None,
                )

            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt, self.config.max_retries, last_error,
                )
                self._backoff(attempt)

        # All retries exhausted
        duration = _now_ms() - start_ms
        logger.error(
            "LLM call failed after %d attempts (last error: %s)",
            self.config.max_retries, last_error,
        )
        return LLMResponse(
            raw_text=raw_text,
            parsed_json=None,
            model=self.config.model,
            duration_ms=duration,
            success=False,
            error=last_error,
        )

    # -- internals --

    def _post(self, body: bytes) -> str:
        """Low-level HTTP POST, returns the model's text response."""
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            req, timeout=self.config.timeout_seconds
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Ollama returns {"response": "...", ...}
        text = data.get("response", "")
        if not text:
            raise ValueError(f"Empty response from LLM: {data}")
        return text

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        base = self.config.retry_base_delay * (2 ** (attempt - 1))
        jitter = base * self.config.retry_jitter * (2 * random.random() - 1)
        delay = max(0, base + jitter)
        logger.debug("Backing off %.1fs before retry", delay)
        time.sleep(delay)

    # -- validation-aware generation --

    def generate_with_validation(
        self,
        prompt: str,
        system: str,
        entity_extractor,
        validator,
        max_validation_retries: int = 2,
        expect_json: bool = True,
    ) -> LLMResponse:
        """Generate, then validate extracted entities.  On validation
        failure, re-prompt the LLM with error feedback.

        Args:
            prompt: The user/instruction prompt.
            system: System prompt.
            entity_extractor: callable(parsed_json) -> list[dict]
            validator: callable(list[dict]) -> list[ValidationResult]
            max_validation_retries: extra LLM calls for fixing failures.
            expect_json: whether to parse JSON from response.

        Returns:
            Best LLMResponse obtained (possibly after retries).
        """
        from validation import build_feedback_prompt

        best_response: LLMResponse | None = None
        best_pass_count = -1
        current_prompt = prompt

        for attempt in range(1 + max_validation_retries):
            resp = self.generate(
                prompt=current_prompt,
                system=system,
                expect_json=expect_json,
            )

            if not resp.success or resp.parsed_json is None:
                # LLM-level failure — return immediately
                return resp

            entities = entity_extractor(resp.parsed_json)
            if not entities:
                return resp  # nothing to validate

            results = validator(entities)
            pass_count = sum(1 for vr in results if vr.passed)

            if pass_count > best_pass_count:
                best_pass_count = pass_count
                best_response = resp

            if all(vr.passed for vr in results):
                logger.debug(
                    "Validation passed on attempt %d/%d",
                    attempt + 1, 1 + max_validation_retries,
                )
                return resp

            if attempt < max_validation_retries:
                feedback = build_feedback_prompt(results, entities)
                current_prompt = prompt + "\n\n" + feedback
                logger.info(
                    "Validation retry %d/%d: %d/%d entities failed",
                    attempt + 1, max_validation_retries,
                    len(results) - pass_count, len(results),
                )

        logger.warning(
            "Validation retries exhausted; returning best response "
            "(%d/%d passed)",
            best_pass_count, len(entities) if entities else 0,
        )
        return best_response or resp

    # -- health check --

    def ping(self) -> bool:
        """Check if the LLM server is reachable."""
        try:
            url = f"{self.config.base_url.rstrip('/')}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenAI-compatible client (Medtronic company GPT)
# ---------------------------------------------------------------------------

@dataclass
class OpenAIConfig:
    """Config for an OpenAI-compatible endpoint (e.g. Medtronic company GPT)."""

    base_url: str = "https://api.gpt.medtronic.com"
    model: str = "gpt-4o"
    subscription_key: str = ""          # subscription-key header
    api_token: str = ""                 # api-token header
    api_version: str = "3.0"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_base_delay: float = 2.0
    retry_jitter: float = 0.5
    temperature: float = 0.1
    max_tokens: int = 4096
    max_prompt_chars: int = _DEFAULT_MAX_PROMPT_CHARS

    def __post_init__(self) -> None:
        if not self.subscription_key:
            raise ValueError("OpenAIConfig.subscription_key must be set")
        if not self.api_token:
            raise ValueError("OpenAIConfig.api_token must be set")
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be > 0")


class OpenAIClient:
    """Drop-in replacement for LLMClient using an OpenAI-compatible endpoint.

    Targets the Medtronic company GPT API:
      POST https://api.gpt.medtronic.com/v1/chat/completions
      Headers: subscription-key, api-token, api-version
    """

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        self._endpoint = f"{config.base_url.rstrip('/')}/models/{config.model}"

    def generate(
        self,
        prompt: str,
        system: str,
        expect_json: bool = True,
    ) -> LLMResponse:
        total_chars = len(prompt) + len(system)
        if total_chars > self.config.max_prompt_chars:
            msg = (
                f"Prompt too large: {total_chars} chars "
                f"(limit {self.config.max_prompt_chars})"
            )
            logger.error(msg)
            raise PromptTooLargeError(msg)

        payload: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        start_ms = _now_ms()

        raw_text = ""
        last_error: str | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_text = self._post(body)
                parsed = extract_json(raw_text) if expect_json else None
                if expect_json and parsed is None:
                    last_error = f"JSON extraction failed on attempt {attempt}"
                    logger.warning(
                        "Attempt %d/%d: non-JSON response (%.100s…), retrying",
                        attempt, self.config.max_retries, raw_text,
                    )
                    self._backoff(attempt)
                    continue
                return LLMResponse(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    model=self.config.model,
                    duration_ms=_now_ms() - start_ms,
                    success=True,
                    error=None,
                )
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt, self.config.max_retries, last_error,
                )
                self._backoff(attempt)

        duration = _now_ms() - start_ms
        logger.error(
            "LLM call failed after %d attempts (last error: %s)",
            self.config.max_retries, last_error,
        )
        return LLMResponse(
            raw_text=raw_text,
            parsed_json=None,
            model=self.config.model,
            duration_ms=duration,
            success=False,
            error=last_error,
        )

    def _post(self, body: bytes) -> str:
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type":     "application/json",
                "subscription-key": self.config.subscription_key.strip(),
                "api-token":        "".join(self.config.api_token.split()),
                "api-version":      self.config.api_version,
            },
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected response shape: {data}") from exc
        if not text:
            raise ValueError(f"Empty content in response: {data}")
        return text

    def _backoff(self, attempt: int) -> None:
        base = self.config.retry_base_delay * (2 ** (attempt - 1))
        jitter = base * self.config.retry_jitter * (2 * random.random() - 1)
        time.sleep(max(0, base + jitter))

    def generate_with_validation(
        self,
        prompt: str,
        system: str,
        entity_extractor,
        validator,
        max_validation_retries: int = 2,
        expect_json: bool = True,
    ) -> LLMResponse:
        """Same interface as LLMClient.generate_with_validation."""
        from validation import build_feedback_prompt

        best_response: LLMResponse | None = None
        best_pass_count = -1
        current_prompt = prompt
        entities: list = []

        for attempt in range(1 + max_validation_retries):
            resp = self.generate(current_prompt, system, expect_json)
            if not resp.success or resp.parsed_json is None:
                return resp
            entities = entity_extractor(resp.parsed_json)
            if not entities:
                return resp
            results = validator(entities)
            pass_count = sum(1 for vr in results if vr.passed)
            if pass_count > best_pass_count:
                best_pass_count = pass_count
                best_response = resp
            if all(vr.passed for vr in results):
                return resp
            if attempt < max_validation_retries:
                current_prompt = prompt + "\n\n" + build_feedback_prompt(results, entities)
                logger.info(
                    "Validation retry %d/%d: %d/%d entities failed",
                    attempt + 1, max_validation_retries,
                    len(results) - pass_count, len(results),
                )

        logger.warning(
            "Validation retries exhausted; returning best response (%d/%d passed)",
            best_pass_count, len(entities),
        )
        return best_response or resp

    def ping(self) -> bool:
        """Always returns True — connectivity is verified on the first real call."""
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)
