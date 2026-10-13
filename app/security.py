"""
Security primitives: constant-time auth, token-bucket rate limiting, security
headers, body limits, and a hardened HTTP client.

Everything here is standard library. No dependency means no supply-chain
surface and no CVE feed to watch for this layer.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Security headers
# --------------------------------------------------------------------------

# Applied to every response. The CSP is strict because the UI is plain
# HTML/CSS/JS served from this same origin with no inline scripts and no CDNs.
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), usb=()",
    "Content-Security-Policy": (
        "default-src 'none'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'; "
        "object-src 'none'"
    ),
}


def security_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = dict(SECURITY_HEADERS)
    if extra:
        h.update(extra)
    return h


# --------------------------------------------------------------------------
# Constant-time authentication
# --------------------------------------------------------------------------

def make_token(nbytes: int = 32) -> str:
    """Generate a strong bearer token. Used by the CLI and docs."""
    return secrets.token_urlsafe(nbytes)


def token_fingerprint(token: str) -> str:
    """Short, non-reversible id for a token. Safe to log."""
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def check_token(supplied: str, expected: str) -> bool:
    """
    Constant-time comparison. Never use == on secrets: it leaks length and
    prefix through timing.
    """
    if not expected:
        return False
    if not supplied:
        return False
    return hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    )


def extract_bearer(header_value: str) -> str:
    """Pull the token out of 'Bearer xyz'. Tolerant of case and spacing."""
    if not header_value:
        return ""
    parts = header_value.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    # Also accept a bare token, which is convenient for curl.
    if len(parts) == 1:
        # ...but a lone "Bearer" with nothing after it is not a token.
        if parts[0].lower() in ("bearer", "token", ""):
            return ""
        return parts[0].strip()
    return ""


# --------------------------------------------------------------------------
# Rate limiting: token bucket per client
# --------------------------------------------------------------------------

@dataclass
class Bucket:
    tokens: float
    last: float


class RateLimiter:
    """
    Per-client token bucket.

    - `rate_per_min` refills continuously.
    - `burst` caps how much credit can accumulate, so an idle client cannot
      bank an unlimited allowance and then hammer.
    - Bounded memory: at most `max_clients` buckets, oldest evicted first.
    """

    def __init__(self, rate_per_min: int, burst: int, max_clients: int = 4096):
        self.rate_per_sec = max(rate_per_min, 1) / 60.0
        self.burst = float(max(burst, 1))
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> tuple[bool, float, float]:
        """
        Consume `cost` from `key`'s bucket.

        Returns (allowed, retry_after_seconds, remaining).
        """
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = Bucket(tokens=self.burst, last=now)
                self._buckets[key] = b
                # Evict oldest beyond the cap.
                while len(self._buckets) > self.max_clients:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(key)

            # Refill.
            elapsed = max(0.0, now - b.last)
            b.tokens = min(self.burst, b.tokens + elapsed * self.rate_per_sec)
            b.last = now

            if b.tokens >= cost:
                b.tokens -= cost
                return True, 0.0, b.tokens

            deficit = cost - b.tokens
            retry_after = deficit / self.rate_per_sec if self.rate_per_sec > 0 else 60.0
            return False, retry_after, b.tokens

    def snapshot(self) -> dict:
        with self._lock:
            return {"clients_tracked": len(self._buckets), "burst": self.burst,
                    "rate_per_min": round(self.rate_per_sec * 60, 2)}


# --------------------------------------------------------------------------
# Client identity
# --------------------------------------------------------------------------

def client_key(ip: str, forwarded_for: str = "", trust_proxy: bool = False) -> str:
    """
    Derive the rate-limit key for a client.

    X-Forwarded-For is attacker-controlled unless you actually sit behind a
    proxy you own, so it is ignored unless TRUST_PROXY is explicitly set.
    """
    if trust_proxy and forwarded_for:
        # Left-most entry is the original client.
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first[:64]
    return ip or "unknown"


# --------------------------------------------------------------------------
# Concurrency limiter (bounds AI calls, protects the box)
# --------------------------------------------------------------------------

class ConcurrencyLimiter:
    """Non-blocking semaphore. try_acquire() fails fast instead of queueing."""

    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self._sem = threading.BoundedSemaphore(self.limit)
        self._in_use = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        if not self._sem.acquire(blocking=False):
            return False
        with self._lock:
            self._in_use += 1
        return True

    def release(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)
        try:
            self._sem.release()
        except ValueError:
            pass

    def stats(self) -> dict:
        with self._lock:
            return {"in_use": self._in_use, "limit": self.limit}


# --------------------------------------------------------------------------
# LRU cache for AI responses
# --------------------------------------------------------------------------

class TTLCache:
    """Small thread-safe LRU with per-entry TTL. Keeps repeat AI calls free."""

    def __init__(self, max_size: int = 512, ttl: int = 900):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Any | None:
        if self.max_size <= 0 or self.ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, value = item
            if expires < now:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        if self.max_size <= 0 or self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


# --------------------------------------------------------------------------
# Circuit breaker: stop hammering a dead endpoint
# --------------------------------------------------------------------------

class CircuitBreaker:
    """
    closed    -> calls pass
    open      -> calls fail instantly until reset_after elapses
    half_open -> one probe call allowed; success closes, failure re-opens
    """

    def __init__(self, fail_threshold: int = 3, reset_after: int = 60):
        self.fail_threshold = max(1, fail_threshold)
        self.reset_after = max(1, reset_after)
        self._failures = 0
        self._opened_at = 0.0
        self._state = "closed"
        self._lock = threading.Lock()

    def _state_locked(self) -> str:
        """
        Read the state assuming the lock is already held.

        Kept separate from the `state` property because threading.Lock is not
        reentrant: calling self.state from inside a locked block deadlocks.
        """
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self.reset_after:
                self._state = "half_open"
        return self._state

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def allow(self) -> bool:
        return self.state in ("closed", "half_open")

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def stats(self) -> dict:
        with self._lock:
            return {
                "state": self._state_locked(),
                "failures": self._failures,
                "threshold": self.fail_threshold,
            }


# --------------------------------------------------------------------------
# Hardened outbound HTTP
# --------------------------------------------------------------------------

def _no_redirect_opener(timeout: int) -> urllib.request.OpenerDirector:
    """
    An opener that refuses redirects.

    Redirects on an AI endpoint are a way to bounce a request somewhere you did
    not validate, so we do not follow them.
    """

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise urllib.error.HTTPError(
                req.full_url, code, f"redirect refused: {newurl}", headers, fp
            )

    ctx = ssl.create_default_context()
    return urllib.request.build_opener(
        NoRedirect,
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.ProxyHandler({}),  # ignore ambient proxy env vars
    )


def safe_request(
    url: str,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    max_bytes: int = 512 * 1024,
) -> tuple[int, bytes]:
    """
    Perform an outbound HTTP request with the guard rails applied:

    - http/https only
    - no redirect following
    - hard timeout on connect and read
    - response size capped, so a hostile endpoint cannot exhaust memory
    - ambient proxy environment variables ignored
    - a User-Agent that identifies this app

    Returns (status, body). Raises on transport errors.
    """
    from config import validate_base_url

    ok, why = validate_base_url(url)
    if not ok:
        raise ValueError(f"refusing to fetch {url}: {why}")

    hdrs = {
        "User-Agent": "treblo-tag-explorer/2.0",
        "Accept": "application/json",
    }
    if headers:
        hdrs.update(headers)

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=hdrs,
                                 method="POST" if data else "GET")
    opener = _no_redirect_opener(timeout)

    with opener.open(req, timeout=timeout) as resp:
        # Read at most max_bytes + 1 so we can detect truncation.
        body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"response exceeded {max_bytes} bytes")
        return resp.status, body


# --------------------------------------------------------------------------
# Input hardening
# --------------------------------------------------------------------------

# Characters that have no business in a search query and are a common sign of
# log-injection or probe attempts.
_CONTROL_CHARS = {chr(c) for c in range(0x00, 0x20)} | {chr(0x7F)}


def sanitize_text(value: str, max_len: int) -> tuple[str, bool]:
    """
    Strip control characters and bound length. Returns (clean, was_modified).

    We do NOT strip quotes or angle brackets: this is a search box and the
    output is escaped at render time. Sanitizing for the wrong context is how
    you mangle legitimate input.
    """
    if not isinstance(value, str):
        return "", True
    original = value
    cleaned = "".join(ch for ch in value if ch not in _CONTROL_CHARS)
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned, cleaned != original


def safe_error(exc: Exception, debug: bool = False) -> str:
    """
    Turn an exception into something safe to show a user.

    Stack traces, file paths, and internal hostnames leak architecture. In
    production the user gets a category and a request id; the detail goes to
    the server log.
    """
    if debug:
        return f"{type(exc).__name__}: {exc}"
    name = type(exc).__name__
    friendly = {
        "FileNotFoundError": "server is missing a required file",
        "PermissionError": "server cannot read a required file",
        "TimeoutError": "operation timed out",
        "ConnectionError": "upstream connection failed",
        "ValueError": "invalid input",
        "KeyError": "missing required field",
        "json.JSONDecodeError": "invalid JSON in request body",
    }
    return friendly.get(name, "internal error")


def request_id() -> str:
    return secrets.token_hex(8)
