"""
AI helpers.

The AI is a helper, never the engine. All real filtering is done by the
deterministic BM25 core; the model only ever chooses from a shortlist the core
already produced, and every value it returns is re-validated against the
vocabulary. A hallucinating model cannot inject a tag that does not exist.

Backends:
  keyless  - free public endpoint. No API key, no signup, no account.
  compat   - anything speaking /v1/chat/completions: OpenAI, Ollama, LM Studio,
             llama.cpp server, vLLM, Groq, Together, OpenRouter. Local = no key.
  off      - no AI at all, zero network calls.

Reliability wrapping, all of it needed because keyless endpoints are shared
best-effort infrastructure:
  - circuit breaker so a dead endpoint fails fast instead of stalling
  - response cache so repeated queries cost nothing
  - concurrency cap so N users cannot spawn N outbound requests
  - hard timeout, response size cap, no redirects, no ambient proxies
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
from dataclasses import dataclass, field
from typing import Any

from security import ConcurrencyLimiter, CircuitBreaker, TTLCache, safe_request
from tags import TagIndex, norm_key

DEFAULT_TIMEOUT = 45

SYSTEM_PROMPT = """You are a music tagging assistant. You turn a plain-English description of
what someone wants to hear into tags from a FIXED vocabulary.

Absolute rules:
1. You may ONLY return tags that appear in the CANDIDATE list you are given.
2. Never invent, translate, pluralize, or reword a tag. Copy it exactly.
3. If nothing fits, return fewer tags. An empty list is a perfectly good answer.
4. Prefer specific tags over generic ones, but never at the cost of rule 1.
5. Return between 0 and {max_tags} tags, ordered most relevant first.

Answer with JSON only. No prose, no markdown fences, exactly this shape:
{{"tags": ["exact tag", "exact tag"], "exclude": ["exact tag"], "reason": "one short sentence"}}"""


@dataclass
class AIResult:
    tags: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    reason: str = ""
    provider: str = ""
    model: str = ""
    ok: bool = False
    error: str = ""
    dropped: list[str] = field(default_factory=list)  # rejected hallucinations
    raw: str = ""
    cached: bool = False
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "used": self.ok,
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "error": self.error,
            "dropped_hallucinations": self.dropped,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
        }


# --------------------------------------------------------------------------
# Response parsing + hallucination guard
# --------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """
    Models love wrapping JSON in prose, in markdown fences, or both.

    Strip fences, try a direct parse, then walk for the first balanced {...}
    block. The brace walker tracks string state so a brace inside a quoted
    reason does not throw off the depth count. Never let json.loads throw on a
    raw model response.
    """
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


def _coerce_list(value: Any) -> list[str]:
    """Accept a string, a list of strings, or a list of objects."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,\n;]+", value) if p.strip()]
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(v.strip())
            elif isinstance(v, dict):
                for k in ("tag", "name", "value", "text"):
                    if isinstance(v.get(k), str):
                        out.append(v[k].strip())
                        break
        return [x for x in out if x]
    return []


def vet(index: TagIndex, wanted: list[str]) -> tuple[list[str], list[str]]:
    """
    Keep only tags that exist in the vocabulary. Return (kept, dropped).

    Matching is case- and accent-insensitive so the model's casing never
    matters, and a unique prefix match is allowed as a last chance so
    "post punk" still resolves to "post-punk".
    """
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in wanted:
        norm = norm_key(raw)
        if not norm:
            continue
        tag = index.get(raw)
        if tag is None:
            cands = [
                t for t in index.tags
                if t.norm.startswith(norm) and len(t.norm) <= len(norm) + 4
            ]
            tag = cands[0] if len(cands) == 1 else None
        if tag is None:
            dropped.append(raw)
            continue
        if tag.norm in seen:
            continue
        seen.add(tag.norm)
        kept.append(tag.name)
    return kept, dropped


# Alias kept because the earlier test suite refers to the private name.
_vet = vet


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class Provider:
    """Base class. Subclasses supply _chat and a default model."""

    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def default_model(self) -> str:
        raise NotImplementedError

    def _chat(self, messages: list[dict], model: str, timeout: int,
              temperature: float) -> str:
        raise NotImplementedError

    def health(self, timeout: int = 12) -> dict:
        raise NotImplementedError


class KeylessProvider(Provider):
    """Free public endpoint. No key, no signup, no account, no billing."""

    name = "keyless"

    def default_model(self) -> str:
        return self.cfg.keyless_model

    def _chat(self, messages, model, timeout, temperature) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": 7,
            "private": True,
        }
        status, body = safe_request(
            self.cfg.keyless_url,
            payload=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        data = json.loads(body.decode("utf-8", "replace"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("no choices in response")
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    def health(self, timeout: int = 12) -> dict:
        try:
            status, body = safe_request(self.cfg.keyless_models_url, timeout=timeout)
            data = json.loads(body.decode("utf-8", "replace"))
            if isinstance(data, list):
                names = [m.get("name") for m in data
                         if isinstance(m, dict) and m.get("name")]
                if names:
                    return {"ok": True, "models": names,
                            "detail": f"{len(names)} keyless model(s) available"}
        except Exception as e:
            return {"ok": False, "models": [], "detail": f"{type(e).__name__}: {e}"}
        return {"ok": False, "models": [],
                "detail": "keyless endpoint unreachable or changed"}


class CompatProvider(Provider):
    """
    Anything speaking /v1/chat/completions. One code path covers OpenAI,
    Ollama, LM Studio, llama.cpp server, vLLM, Groq, Together and OpenRouter.
    """

    name = "compat"

    def default_model(self) -> str:
        return self.cfg.openai_model

    def _chat(self, messages, model, timeout, temperature) -> str:
        url = self.cfg.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Accept": "application/json"}
        if self.cfg.openai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.openai_api_key}"
        payload = {"model": model, "messages": messages, "temperature": temperature}
        status, body = safe_request(url, payload=payload, headers=headers, timeout=timeout)
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        data = json.loads(body.decode("utf-8", "replace"))
        return (data["choices"][0]["message"].get("content") or "").strip()

    def health(self, timeout: int = 12) -> dict:
        base = self.cfg.openai_base_url.rstrip("/")
        try:
            headers = {"Accept": "application/json"}
            if self.cfg.openai_api_key:
                headers["Authorization"] = f"Bearer {self.cfg.openai_api_key}"
            status, body = safe_request(base + "/models", headers=headers, timeout=timeout)
            data = json.loads(body.decode("utf-8", "replace"))
            names = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            return {"ok": True, "models": names[:20], "detail": f"reachable at {base}"}
        except Exception as e:
            return {"ok": False, "models": [],
                    "detail": f"{base} unreachable: {type(e).__name__}"}


def make_provider(cfg) -> Provider | None:
    if not cfg.ai_active:
        return None
    if cfg.ai_provider == "keyless":
        return KeylessProvider(cfg)
    if cfg.ai_provider == "compat":
        return CompatProvider(cfg)
    return None


# --------------------------------------------------------------------------
# The service: a provider plus the reliability machinery
# --------------------------------------------------------------------------

class AIService:
    """
    Owns one provider plus the breaker, cache and concurrency limiter.

    Every public method returns a result object and never raises, so a caller
    can always fall back to engine results.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = make_provider(cfg)
        self.breaker = CircuitBreaker(
            cfg.circuit_fail_threshold, cfg.circuit_reset_seconds
        )
        self.cache = TTLCache(cfg.ai_cache_size, cfg.ai_cache_ttl)
        self.limiter = ConcurrencyLimiter(cfg.ai_max_concurrent)
        self._health_cache: dict = {}
        self._health_at = 0.0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.provider is not None and self.cfg.ai_active

    # ------------------------------------------------------------------

    def interpret(
        self,
        index: TagIndex,
        query: str,
        candidates: list[str],
        model: str = "",
        max_tags: int = 10,
        timeout: int | None = None,
        temperature: float | None = None,
        extra_context: str = "",
    ) -> AIResult:
        """
        Ask the model to pick tags from `candidates` for `query`.

        Returns AIResult, never raises. On any failure `ok` is False and
        `error` explains why, so the caller degrades to engine results.
        """
        provider_name = self.provider.name if self.provider else "off"

        if not self.available:
            return AIResult(provider=provider_name, ok=False, error="ai disabled")
        if not candidates:
            return AIResult(provider=provider_name, ok=False, error="no candidates")

        model = (model or "").strip() or self.provider.default_model()
        timeout = timeout or self.cfg.ai_timeout
        temperature = self.cfg.ai_temperature if temperature is None else temperature

        res = AIResult(provider=provider_name, model=model)

        # --- circuit breaker: fail fast when the endpoint is known dead
        if not self.breaker.allow():
            res.error = "ai temporarily disabled after repeated failures"
            return res

        # --- cache: identical query + candidate set + model is free to repeat
        cache_key = TTLCache.key_for(
            provider_name, model, query, str(max_tags), "|".join(candidates)
        )
        hit = self.cache.get(cache_key)
        if hit is not None:
            cached = AIResult(**hit)
            cached.cached = True
            return cached

        # --- concurrency cap: refuse rather than queue outbound requests
        if not self.limiter.try_acquire():
            res.error = "ai busy, too many concurrent requests"
            return res

        try:
            import time
            t0 = time.monotonic()

            user_prompt = (
                f'What the user wants to hear: "{query}"\n\n'
                "CANDIDATE tags (the only tags you may use):\n"
                + "\n".join(f"- {c}" for c in candidates)
            )
            if extra_context:
                user_prompt += f"\n\nExtra context:\n{extra_context}"
            user_prompt += (
                f"\n\nPick up to {max_tags} of those candidate tags that best match, "
                "plus any you think should be EXCLUDED. JSON only."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(max_tags=max_tags)},
                {"role": "user", "content": user_prompt},
            ]

            raw = self.provider._chat(messages, model, timeout, temperature)
            res.latency_ms = int((time.monotonic() - t0) * 1000)

        except urllib.error.HTTPError as e:
            self.breaker.record_failure()
            res.error = f"HTTP {e.code}"
            return res
        except urllib.error.URLError as e:
            self.breaker.record_failure()
            res.error = f"network: {getattr(e, 'reason', e)}"
            return res
        except TimeoutError:
            self.breaker.record_failure()
            res.error = "timed out"
            return res
        except Exception as e:
            self.breaker.record_failure()
            res.error = f"{type(e).__name__}: {e}"
            return res
        finally:
            self.limiter.release()

        # --- parse and validate
        res.raw = (raw or "")[:2000]
        parsed = _extract_json(raw)
        if not parsed:
            self.breaker.record_failure()
            res.error = "model did not return parseable JSON"
            return res

        # Validate the shape, not just the parse: valid JSON with the wrong
        # keys is a failure, not a success.
        if "tags" not in parsed and "exclude" not in parsed:
            self.breaker.record_failure()
            res.error = "model returned JSON with the wrong shape"
            return res

        kept, dropped = vet(index, _coerce_list(parsed.get("tags")))
        exc, exc_dropped = vet(index, _coerce_list(parsed.get("exclude")))

        res.tags = kept[:max_tags]
        res.exclude = exc
        res.dropped = dropped + exc_dropped
        reason = parsed.get("reason")
        res.reason = reason.strip()[:300] if isinstance(reason, str) else ""
        res.ok = True

        self.breaker.record_success()
        self.cache.put(cache_key, dict(res.__dict__))
        return res

    # ------------------------------------------------------------------

    def health(self, timeout: int = 12, max_age: int = 60) -> dict:
        """Cached reachability probe for the UI's status light."""
        import time

        if not self.available:
            return {"ok": False, "models": [], "detail": "ai disabled",
                    "provider": "off", "circuit": "n/a"}

        with self._lock:
            now = time.monotonic()
            if self._health_cache and now - self._health_at < max_age:
                return self._health_cache

        try:
            result = self.provider.health(timeout=timeout)
        except Exception as e:
            result = {"ok": False, "models": [], "detail": f"{type(e).__name__}: {e}"}

        result["provider"] = self.provider.name
        result["circuit"] = self.breaker.state

        with self._lock:
            self._health_cache = result
            self._health_at = time.monotonic()
        return result

    def stats(self) -> dict:
        return {
            "available": self.available,
            "provider": self.provider.name if self.provider else "off",
            "circuit": self.breaker.stats(),
            "cache": self.cache.stats(),
            "concurrency": self.limiter.stats(),
        }


# --------------------------------------------------------------------------
# Convenience functions for the CLI and tests
# --------------------------------------------------------------------------

def interpret(index, query, candidates, provider="", model="", max_tags=10,
              timeout=DEFAULT_TIMEOUT, temperature=0.3, extra_context="",
              cfg=None) -> AIResult:
    from config import Config

    cfg = cfg or Config.from_env()
    if provider and provider != cfg.ai_provider:
        cfg.ai_provider = provider
    return AIService(cfg).interpret(
        index, query, candidates, model=model, max_tags=max_tags,
        timeout=timeout, temperature=temperature, extra_context=extra_context,
    )


def health(provider: str = "", timeout: int = 12) -> dict:
    from config import Config

    cfg = Config.from_env()
    if provider:
        cfg.ai_provider = provider
    return AIService(cfg).health(timeout=timeout)


def list_keyless_models(timeout: int = 15) -> list[dict]:
    """Ask the keyless service which models it currently offers. No key needed."""
    from config import Config

    cfg = Config.from_env()
    try:
        status, body = safe_request(cfg.keyless_models_url, timeout=timeout)
        data = json.loads(body.decode("utf-8", "replace"))
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and m.get("name")]
    except Exception:
        pass
    return []
