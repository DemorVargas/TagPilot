"""
Central configuration.

Every knob is an environment variable, validated once at startup. Invalid
config fails loudly at boot rather than silently at request time.

Security-relevant defaults are the safe ones: no auth token means "loopback
only", not "wide open".
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        raise SystemExit(f"config error: {name} must be an integer, got {raw!r}")
    if not (lo <= val <= hi):
        raise SystemExit(f"config error: {name} must be between {lo} and {hi}, got {val}")
    return val


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        raise SystemExit(f"config error: {name} must be a number, got {raw!r}")
    if not (lo <= val <= hi):
        raise SystemExit(f"config error: {name} must be between {lo} and {hi}, got {val}")
    return val


# --------------------------------------------------------------------------
# SSRF guard
# --------------------------------------------------------------------------

# Addresses that must never be reachable through a user-influenced base URL:
# cloud metadata services and link-local space. Loopback and RFC1918 stay
# allowed because "local Ollama on the host" is a supported, intended mode.
BLOCKED_HOSTS = {
    "169.254.169.254",       # AWS/GCP/Azure metadata
    "metadata.google.internal",
    "100.100.100.200",       # Alibaba metadata
    "fd00:ec2::254",         # AWS metadata over IPv6
}


def validate_base_url(url: str) -> tuple[bool, str]:
    """
    Validate an OpenAI-compatible base URL.

    Returns (ok, reason). Rejects non-http(s) schemes (file://, gopher://,
    dict://) and known cloud metadata endpoints. Allows loopback and private
    ranges on purpose, because local inference is a first-class feature.
    """
    if not url:
        return False, "empty url"
    try:
        p = urlparse(url)
    except ValueError as e:
        return False, f"unparseable: {e}"

    if p.scheme not in ("http", "https"):
        return False, f"scheme must be http or https, got {p.scheme!r}"
    if not p.hostname:
        return False, "no host in url"

    host = p.hostname.lower()
    if host in BLOCKED_HOSTS:
        return False, f"{host} is a blocked metadata endpoint"

    # Resolve-ish check: if it is a literal IP, verify it is not link-local.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_link_local:
            return False, f"{host} is link-local and blocked"
    except ValueError:
        pass  # a hostname, fine

    return True, "ok"


# --------------------------------------------------------------------------

@dataclass
class Config:
    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    tag_file: Path = ROOT / "data" / "treblo_tags_v3.txt"
    web_dir: Path = ROOT / "web"
    verbose: bool = False
    debug: bool = False

    # --- runtime mode: online | offline | hybrid ---
    mode: str = "hybrid"

    # --- security ---
    auth_token: str = ""
    auth_required: bool = False
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 15
    ai_rate_limit_per_min: int = 12
    max_threads: int = 24
    max_body_bytes: int = 64 * 1024
    max_query_chars: int = 500
    max_url_chars: int = 2048
    socket_timeout: int = 30
    trust_proxy: bool = False
    allowed_origins: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)
    tls_cert: str = ""
    tls_key: str = ""

    # --- AI ---
    ai_enabled: bool = True
    ai_provider: str = "keyless"
    keyless_model: str = "openai-fast"
    keyless_url: str = "https://text.pollinations.ai/openai"
    keyless_models_url: str = "https://text.pollinations.ai/models"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    openai_model: str = "llama3.2"
    ai_timeout: int = 45
    ai_temperature: float = 0.3
    ai_max_concurrent: int = 4
    ai_cache_ttl: int = 900
    ai_cache_size: int = 512
    circuit_fail_threshold: int = 3
    circuit_reset_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.host = os.environ.get("HOST", c.host)
        c.port = _int("PORT", c.port, 1, 65535)
        c.tag_file = Path(os.environ.get("TAG_FILE", str(c.tag_file))).expanduser()
        c.web_dir = Path(os.environ.get("WEB_DIR", str(c.web_dir))).expanduser()
        c.verbose = _bool("VERBOSE", False)
        c.debug = _bool("DEBUG", False)

        c.mode = os.environ.get("MODE", c.mode).strip().lower()
        if c.mode not in ("online", "offline", "hybrid"):
            raise SystemExit(f"config error: MODE must be online|offline|hybrid, got {c.mode!r}")

        # --- security ---
        c.auth_token = os.environ.get("AUTH_TOKEN", "").strip()
        c.auth_required = _bool("AUTH_REQUIRED", bool(c.auth_token))
        if c.auth_required and not c.auth_token:
            raise SystemExit(
                "config error: AUTH_REQUIRED=1 but AUTH_TOKEN is empty. "
                "Set AUTH_TOKEN or turn auth off."
            )
        if c.auth_token and len(c.auth_token) < 16:
            raise SystemExit("config error: AUTH_TOKEN must be at least 16 characters")

        c.rate_limit_per_min = _int("RATE_LIMIT_PER_MIN", c.rate_limit_per_min, 1, 100000)
        c.rate_limit_burst = _int("RATE_LIMIT_BURST", c.rate_limit_burst, 1, 10000)
        c.ai_rate_limit_per_min = _int("AI_RATE_LIMIT_PER_MIN", c.ai_rate_limit_per_min, 1, 10000)
        c.max_threads = _int("MAX_THREADS", c.max_threads, 2, 512)
        c.max_body_bytes = _int("MAX_BODY_BYTES", c.max_body_bytes, 1024, 10_000_000)
        c.max_query_chars = _int("MAX_QUERY_CHARS", c.max_query_chars, 16, 10000)
        c.max_url_chars = _int("MAX_URL_CHARS", c.max_url_chars, 64, 65536)
        c.socket_timeout = _int("SOCKET_TIMEOUT", c.socket_timeout, 5, 600)
        c.trust_proxy = _bool("TRUST_PROXY", False)
        c.tls_cert = os.environ.get("TLS_CERT", "").strip()
        c.tls_key = os.environ.get("TLS_KEY", "").strip()
        origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
        c.allowed_origins = [o.strip() for o in origins.split(",") if o.strip()]
        # Extra names this instance is legitimately reached by (a LAN
        # hostname, a reverse-proxy domain). Anything not listed here and not
        # a local address is treated as a foreign origin and refused.
        hosts_env = os.environ.get("ALLOWED_HOSTS", "").strip()
        c.allowed_hosts = [h.strip().lower() for h in hosts_env.split(",") if h.strip()]

        # --- AI ---
        c.ai_enabled = _bool("AI_ENABLED", True)
        c.ai_provider = os.environ.get("AI_PROVIDER", c.ai_provider).strip().lower()
        if c.ai_provider not in ("keyless", "compat", "off"):
            raise SystemExit(
                f"config error: AI_PROVIDER must be keyless|compat|off, got {c.ai_provider!r}"
            )
        c.keyless_model = os.environ.get("KEYLESS_MODEL", c.keyless_model)
        c.keyless_url = os.environ.get("KEYLESS_URL", c.keyless_url)
        c.keyless_models_url = os.environ.get("KEYLESS_MODELS_URL", c.keyless_models_url)
        c.openai_base_url = os.environ.get("OPENAI_BASE_URL", c.openai_base_url).rstrip("/")
        c.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        c.openai_model = os.environ.get("OPENAI_MODEL", c.openai_model)
        c.ai_timeout = _int("AI_TIMEOUT", c.ai_timeout, 3, 300)
        c.ai_temperature = _float("AI_TEMPERATURE", c.ai_temperature, 0.0, 2.0)
        c.ai_max_concurrent = _int("AI_MAX_CONCURRENT", c.ai_max_concurrent, 1, 64)
        c.ai_cache_ttl = _int("AI_CACHE_TTL", c.ai_cache_ttl, 0, 86400)
        c.ai_cache_size = _int("AI_CACHE_SIZE", c.ai_cache_size, 0, 100000)
        c.circuit_fail_threshold = _int("CIRCUIT_FAIL_THRESHOLD", c.circuit_fail_threshold, 1, 100)
        c.circuit_reset_seconds = _int("CIRCUIT_RESET_SECONDS", c.circuit_reset_seconds, 1, 3600)

        # --- mode coherence ---
        if c.mode == "offline":
            c.ai_enabled = False
        if c.mode == "online" and not c.ai_enabled:
            # explicit online with AI off is contradictory; honour AI off
            c.mode = "offline"

        # --- validate the outbound URL we will actually call ---
        if c.ai_enabled and c.ai_provider == "compat":
            ok, why = validate_base_url(c.openai_base_url)
            if not ok:
                raise SystemExit(f"config error: OPENAI_BASE_URL rejected: {why}")
        if c.ai_enabled and c.ai_provider == "keyless":
            ok, why = validate_base_url(c.keyless_url)
            if not ok:
                raise SystemExit(f"config error: KEYLESS_URL rejected: {why}")

        if bool(c.tls_cert) != bool(c.tls_key):
            raise SystemExit("config error: TLS_CERT and TLS_KEY must be set together")

        return c

    # ----------------------------------------------------------------------

    @property
    def ai_active(self) -> bool:
        """AI is usable only when enabled, provider set, and mode permits it."""
        return self.ai_enabled and self.ai_provider != "off" and self.mode != "offline"

    @property
    def loopback_only(self) -> bool:
        """When no token is set we still refuse to be an open relay."""
        return not self.auth_required

    def public_dict(self) -> dict:
        """Safe to hand to the browser. Never includes secrets."""
        return {
            "mode": self.mode,
            "ai_enabled": self.ai_enabled,
            "ai_provider": self.ai_provider if self.ai_active else "off",
            "auth_required": self.auth_required,
            "rate_limit_per_min": self.rate_limit_per_min,
            "ai_rate_limit_per_min": self.ai_rate_limit_per_min,
            "tls": bool(self.tls_cert),
            "max_query_chars": self.max_query_chars,
            "version": VERSION,
        }

    def redacted_summary(self) -> dict:
        """Startup banner. Shows that a secret exists, never the secret."""
        return {
            "mode": self.mode,
            "host": self.host,
            "port": self.port,
            "ai_active": self.ai_active,
            "ai_provider": self.ai_provider,
            "api_key_set": bool(self.openai_api_key),
            "auth": "token required" if self.auth_required else "none (bind to loopback!)",
            "rate_limit": f"{self.rate_limit_per_min}/min burst {self.rate_limit_burst}",
            "tls": "on" if self.tls_cert else "off",
            "max_threads": self.max_threads,
        }


VERSION = "2.0.0"
