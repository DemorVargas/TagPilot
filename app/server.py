"""
Treblo Tag Explorer - HTTP server.

Standard library only. No Flask, no FastAPI, no build step, no npm.
Serves the web UI and a JSON API.

Run:  python3 app/server.py
Env:  see app/config.py - every knob is documented there.

Security posture, in one place so it can be reviewed:
  - optional bearer-token auth, compared in constant time
  - when no token is set, binds loopback only (never an open relay)
  - per-client token-bucket rate limiting, stricter for AI routes
  - bounded thread pool, so a flood cannot exhaust the box
  - request body size cap, query length cap, control characters stripped
  - strict CSP and the usual hardening headers on every response
  - outbound requests are http(s) only, no redirects, size-capped, timeout
  - internal errors are logged, never returned to the client
  - path traversal is structurally impossible (no user input reaches a path)
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import ai as ai_mod  # noqa: E402
import pipeline  # noqa: E402
from config import Config  # noqa: E402
from security import (  # noqa: E402
    RateLimiter,
    check_token,
    client_key,
    extract_bearer,
    request_id,
    safe_error,
    sanitize_text,
    security_headers,
    token_fingerprint,
)
from tags import CATEGORIES, TagIndex  # noqa: E402

# --------------------------------------------------------------------------
# Process-wide state
# --------------------------------------------------------------------------

CFG: Config = Config.from_env()

_index: TagIndex | None = None
_index_lock = threading.Lock()

_ai_service: ai_mod.AIService | None = None
_ai_lock = threading.Lock()

# Separate limiters: the AI limiter is deliberately much stricter, because
# each AI request is a slow outbound call that costs someone else money.
_general_limiter = RateLimiter(CFG.rate_limit_per_min, CFG.rate_limit_burst)
_ai_limiter = RateLimiter(CFG.ai_rate_limit_per_min, max(1, CFG.rate_limit_burst // 2))

# Static file table. A fixed dict means no user input ever reaches a path.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def get_index() -> TagIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                if not CFG.tag_file.exists():
                    raise FileNotFoundError(f"tag file not found: {CFG.tag_file}")
                _index = TagIndex(CFG.tag_file)
    return _index


_local_addr_cache: set[str] | None = None


def _local_addresses() -> set[str]:
    """
    Every address this machine currently holds, so a request arriving by any of
    them counts as same-origin.

    Cached after the first call: interfaces change rarely and a socket lookup
    per request would be wasteful. Restart if you add an interface.

    The UDP connect() trick is the standard way to learn the primary outbound
    address without sending a packet. Failures are ignored on purpose: a
    machine with no network should still serve loopback.
    """
    global _local_addr_cache
    if _local_addr_cache is not None:
        return _local_addr_cache

    found: set[str] = set()

    # gethostbyname_ex covers the usual case without shelling out.
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        found |= {a for a in addrs if a}
    except OSError:
        pass

    # Primary outbound address.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            found.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass

    # Hostname aliases, so a machine reachable as "mybox.local" works too.
    try:
        name = socket.gethostname()
        if name:
            found.add(name)
            found.add(name.split(".")[0])
    except OSError:
        pass

    _local_addr_cache = {f.lower() for f in found if f}
    return _local_addr_cache


def get_ai_service() -> ai_mod.AIService:
    global _ai_service
    if _ai_service is None:
        with _ai_lock:
            if _ai_service is None:
                _ai_service = ai_mod.AIService(CFG)
    return _ai_service


# --------------------------------------------------------------------------
# Server with a bounded thread pool
# --------------------------------------------------------------------------

class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """
    ThreadingHTTPServer spawns an unbounded thread per connection, which is a
    trivial denial of service on a small box. A semaphore caps concurrency.

    Important: a slot is held for the lifetime of the *connection*, not the
    request, and HTTP/1.1 keep-alive means browsers hold connections open. So
    the handler sets a short idle timeout (see Handler.timeout) and idle
    connections are closed to release their slot. Without that, a handful of
    browser tabs can exhaust the pool and every later request is refused.
    """

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, *args, max_threads: int = 24, **kwargs):
        self.max_threads = max(1, max_threads)
        self._pool = threading.BoundedSemaphore(self.max_threads)
        super().__init__(*args, **kwargs)

    def _reject(self, request) -> None:
        """
        Refuse a connection with a valid JSON error body.

        Returning an empty body here is what made the browser report
        "JSON.parse: unexpected character at line 1 column 1" instead of a
        useful message: the client called response.json() on nothing.
        """
        payload = json.dumps({
            "error": "server busy, too many concurrent connections",
            "detail": f"limit is {self.max_threads}; retry shortly",
        }).encode("utf-8")
        head = (
            "HTTP/1.1 503 Service Unavailable\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Retry-After: 1\r\n"
            "Connection: close\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "\r\n"
        ).encode("ascii")
        try:
            request.sendall(head + payload)
        except OSError:
            pass

    def process_request(self, request, client_address):
        # Non-blocking acquire: refuse rather than queue without bound.
        if not self._pool.acquire(blocking=False):
            self._reject(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._pool.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._pool.release()


# --------------------------------------------------------------------------
# Request handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "TrebloTagExplorer"
    sys_version = ""          # do not advertise the Python version
    protocol_version = "HTTP/1.1"

    # HTTP/1.1 keep-alive holds a worker thread (and a pool slot) per open
    # connection. Browsers keep connections idle for a long time, so without a
    # short idle timeout a few tabs exhaust the thread pool and every later
    # request gets refused. The timeout applies to the *next* request read on a
    # kept-alive connection: when it expires the handler returns and the slot
    # is released.
    timeout = 15

    def setup(self) -> None:
        # BaseHTTPRequestHandler does not apply its `timeout` class attribute
        # to the socket by itself; StreamRequestHandler does, but only when
        # timeout is set before setup runs. Set it explicitly to be sure.
        super().setup()
        if self.timeout is not None:
            try:
                self.connection.settimeout(self.timeout)
            except OSError:
                pass

    # ---------- response helpers ----------

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        # Drain any unread request body BEFORE responding. Leaving it in the
        # socket corrupts the next request on a keep-alive connection.
        self._drain_if_needed()
        hdrs = security_headers(extra)
        if self.close_connection:
            hdrs["Connection"] = "close"
        # Only an explicitly allow-listed cross-origin caller gets CORS
        # headers. Same-origin callers need none, and everyone else gets none
        # on purpose so the browser blocks the response.
        echo = getattr(self, "_cors_echo", None)
        if echo:
            hdrs["Access-Control-Allow-Origin"] = echo
            hdrs["Vary"] = "Origin"
            hdrs["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            hdrs["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            hdrs["Access-Control-Max-Age"] = "600"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def _json(self, obj, code: int = 200, extra: dict[str, str] | None = None) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8", extra)

    def _err(self, code: int, msg: str, rid: str = "") -> None:
        payload = {"error": msg}
        if rid:
            payload["request_id"] = rid
        self._json(payload, code)

    # ---------- request helpers ----------

    def _body(self) -> tuple[dict, str]:
        """
        Read and parse the JSON body under a size cap.

        Returns (obj, error). A body over the cap is rejected without reading
        it into memory, so a huge Content-Length cannot exhaust RAM. Records
        how many bytes were consumed so the drain step knows what is left.
        """
        self._body_consumed = 0
        raw_len = self.headers.get("Content-Length")
        try:
            n = int(raw_len or 0)
        except (TypeError, ValueError):
            return {}, "invalid Content-Length"
        if n < 0:
            return {}, "invalid Content-Length"
        if n > CFG.max_body_bytes:
            # Deliberately not read: the drain step closes the connection
            # instead, because swallowing an oversized body is a DoS vector.
            return {}, f"body too large (max {CFG.max_body_bytes} bytes)"
        if n == 0:
            return {}, ""
        try:
            data = self.rfile.read(n)
            self._body_consumed = len(data)
        except OSError:
            return {}, "could not read body"
        try:
            obj = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}, "body must be valid UTF-8 JSON"
        if not isinstance(obj, dict):
            return {}, "body must be a JSON object"
        return obj, ""

    def _client_ip(self) -> str:
        try:
            return self.client_address[0]
        except (IndexError, TypeError):
            return "unknown"

    # ---------- keep-alive correctness ----------

    # Bodies larger than this are never drained: the connection is closed
    # instead. Draining a hostile 10GB body to preserve keep-alive would be a
    # self-inflicted DoS.
    DRAIN_LIMIT = 1_000_000

    def _drain_if_needed(self) -> None:
        """
        Drain only when the body has not already been read by the handler.

        `_body()` records how much it consumed. If it consumed the declared
        length there is nothing left, and reading again would block until the
        idle timeout. If the handler never touched the body, `_body_consumed`
        is None and the whole declared length must be discarded.
        """
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return

        consumed = getattr(self, "_body_consumed", None)
        if consumed is None:
            remaining = length
        else:
            remaining = max(0, length - int(consumed))

        if remaining <= 0:
            return
        if remaining > self.DRAIN_LIMIT:
            self.close_connection = True
            return
        try:
            self.rfile.read(remaining)
            self._body_consumed = (consumed or 0) + remaining
        except (OSError, ValueError):
            self.close_connection = True

    # ---------- error responses in the client's format ----------

    def send_error(self, code, message=None, explain=None):
        """
        Override BaseHTTPRequestHandler's error path.

        The default emits an HTML page, which is wrong for a JSON API and is
        exactly what produced "server returned an HTML page (400)". Protocol
        level errors (malformed request line, unsupported method, headers too
        long) now come back as JSON too, so a client always gets something it
        can parse.
        """
        short, long = self.responses.get(code, ("Error", "Error"))
        try:
            payload = json.dumps({
                "error": str(message) if message else short,
                "detail": str(explain) if explain else long,
                "status": int(code),
            }).encode("utf-8")
        except (TypeError, ValueError):
            payload = b'{"error":"request failed"}'

        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in security_headers().items():
                self.send_header(k, v)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _rate_key(self, kind: str) -> str:
        ip = self._client_ip()
        fwd = self.headers.get("X-Forwarded-For", "")
        return kind + ":" + client_key(ip, fwd, CFG.trust_proxy)

    def _limit(self, kind: str = "general", cost: float = 1.0) -> bool:
        """
        Apply the relevant limiter. On refusal sends 429 and returns False.
        """
        limiter = _ai_limiter if kind == "ai" else _general_limiter
        allowed, retry_after, _ = limiter.check(self._rate_key(kind), cost)
        if allowed:
            return True
        self._json(
            {"error": "rate limit exceeded", "retry_after_seconds": round(retry_after, 1)},
            429,
            extra={"Retry-After": str(max(1, int(retry_after)))},
        )
        return False

    def _authorized(self) -> bool:
        """
        Check the bearer token when auth is on.

        With no token configured we allow the request: the server is expected
        to be bound to loopback in that configuration, and startup warns about
        it. That is a deliberate choice for the single-user Docker case, and
        the alternative (refusing everything) makes the default install
        unusable.
        """
        if not CFG.auth_required:
            return True
        supplied = extract_bearer(self.headers.get("Authorization", ""))
        if check_token(supplied, CFG.auth_token):
            return True
        self._json(
            {"error": "unauthorized"},
            401,
            extra={"WWW-Authenticate": 'Bearer realm="treblo-tag-explorer"'},
        )
        return False

    def _origin_hosts(self) -> set[str]:
        """
        Hostnames that count as "this server".

        Two jobs, and both matter:

        1. Allow whichever name the user actually reached us by. That could be
           loopback, a LAN address, a hostname, or a reverse-proxy domain, and
           it differs per client.
        2. Refuse anything else, which is the DNS-rebinding defence. Without a
           known-name check, an attacker's domain that resolves to this
           machine would arrive with Origin and Host both set to the attacker's
           domain, matching each other and sailing through. Checking the name
           against a known set closes that.

        Populated from: loopback aliases, this machine's own addresses, the
        configured bind host, ALLOWED_HOSTS, and the hosts in ALLOWED_ORIGINS.
        """
        cached = getattr(self, "_origin_hosts_cache", None)
        if cached is not None:
            return cached

        hosts = {"localhost", "127.0.0.1", "::1", "[::1]"}
        hosts |= _local_addresses()

        if CFG.host and CFG.host not in ("0.0.0.0", "::", ""):
            hosts.add(CFG.host)
        hosts |= {h.lower() for h in CFG.allowed_hosts}

        # A cross-origin grant implies that host is ours to serve.
        for extra in CFG.allowed_origins:
            try:
                p = urlparse(extra)
                if p.hostname:
                    hosts.add(p.hostname.lower())
            except ValueError:
                pass

        self._origin_hosts_cache = hosts
        return hosts

    def _same_origin(self, origin: str) -> bool:
        """
        Is this Origin the same server we are running on?

        Same-origin means the Origin matches the host the request actually
        arrived on. Compared against the Host header, not against a hardcoded
        address, so it works whether the user reached us at localhost, a LAN
        IP, a machine hostname, or through a reverse proxy.

        Both names must also be in the known set (see _origin_hosts), which is
        what stops DNS rebinding.
        """
        try:
            o = urlparse(origin)
        except ValueError:
            return False
        if not o.hostname:
            return False

        host_header = self.headers.get("Host", "")
        if not host_header:
            return False
        try:
            h = urlparse("//" + host_header)
        except ValueError:
            return False
        if not h.hostname:
            return False

        known = self._origin_hosts()
        o_host = o.hostname.lower()
        h_host = h.hostname.lower()

        # Both sides must be a name we recognise, and they must agree.
        if o_host not in known or h_host not in known:
            return False
        if o_host != h_host:
            return False

        # Ports: only compare when both sides state one explicitly. Behind a
        # TLS-terminating proxy the Origin says 443 implicitly while the Host
        # header often carries no port at all; requiring equality there would
        # reject a perfectly valid same-origin request.
        if o.port is not None and h.port is not None:
            return o.port == h.port
        return True

    def _cors(self) -> bool:
        """
        Decide whether to serve a request that carries an Origin header.

        CORS is about CROSS-origin access, not same-origin. Browsers send an
        `Origin` header on same-origin POSTs too, so rejecting anything not in
        an explicit allow-list breaks the app's own UI. Treating same-origin as
        forbidden was a bug.

        Rules:
          - no Origin header (curl, server-to-server): allow
          - Origin is this same server: allow
          - Origin is in ALLOWED_ORIGINS: allow, and echo it back
          - anything else: refuse

        On refusal the response carries no CORS headers, so the browser blocks
        it as intended.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        if self._same_origin(origin):
            self._cors_echo = None      # same-origin: no CORS headers needed
            return True
        if origin in CFG.allowed_origins:
            self._cors_echo = origin    # explicit cross-origin grant
            return True
        self._cors_echo = None
        return False

    def log_message(self, fmt, *args) -> None:
        if CFG.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def log_error(self, fmt, *args) -> None:
        # Always surface server-side errors, even when quiet.
        sys.stderr.write("server: %s\n" % (fmt % args))

    def handle_one_request(self) -> None:
        """
        Wrap request handling so keep-alive state cannot leak between requests.

        Two things matter here:

        1. `_body_consumed` is reset per request. Without that, a request with
           no body inherits the previous request's count and the drain step
           misjudges how much is left in the socket.
        2. If a handler raises before responding, any unread body is drained
           (or the connection closed) before the error response goes out.
           Otherwise the leftover bytes corrupt the next request.

        Tracebacks from clients hanging up, or from an idle keep-alive
        connection timing out, are suppressed: both are normal traffic.
        """
        self._body_consumed = None
        self._cors_echo = None
        self._origin_hosts_cache = None
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except TimeoutError:
            self.close_connection = True
        except OSError:
            self.close_connection = True
        except Exception as exc:
            # A bug in a handler must not leave the socket in a corrupt state.
            sys.stderr.write(
                f"handler error: {type(exc).__name__}: {exc}\n"
            )
            try:
                self._drain_if_needed()
            except Exception:
                self.close_connection = True
            self.close_connection = True

    # ---------- routing ----------

    def do_GET(self):
        self._route("GET")

    def do_HEAD(self):
        self._route("HEAD")

    def do_POST(self):
        self._route("POST")

    def do_OPTIONS(self):
        """
        Preflight. Only an allow-listed cross-origin caller gets a real answer;
        same-origin callers never preflight, and everyone else is refused so the
        browser blocks the actual request.
        """
        origin = self.headers.get("Origin", "")
        if origin and origin in CFG.allowed_origins:
            self._cors_echo = origin
            return self._send(204, b"", "text/plain; charset=utf-8",
                              extra={"Allow": "GET, POST, OPTIONS"})
        if origin and self._same_origin(origin):
            return self._send(204, b"", "text/plain; charset=utf-8",
                              extra={"Allow": "GET, POST, OPTIONS"})
        return self._err(403, "origin not allowed")

    def _route(self, method: str) -> None:
        rid = request_id()
        try:
            if len(self.path) > CFG.max_url_chars:
                return self._err(414, "URL too long", rid)

            u = urlparse(self.path)
            path, qs = u.path, parse_qs(u.query, max_num_fields=32)

            # Static assets: no auth, no rate limit, no user input in paths.
            if method in ("GET", "HEAD") and path in STATIC:
                name, ctype = STATIC[path]
                return self._static(name, ctype)

            if not self._cors():
                return self._err(403, "origin not allowed", rid)

            if not self._authorized():
                return None

            if method in ("GET", "HEAD"):
                return self._get(path, qs, rid)
            if method == "POST":
                return self._post(path, rid)
            return self._err(405, "method not allowed", rid)

        except BrokenPipeError:
            pass
        except Exception as exc:  # never leak internals to the client
            sys.stderr.write(f"[{rid}] unhandled: {type(exc).__name__}: {exc}\n")
            self._err(500, safe_error(exc, CFG.debug), rid)

    def _static(self, name: str, ctype: str) -> None:
        # `name` comes from the STATIC table, never from the request, so there
        # is no path to traverse.
        p = CFG.web_dir / name
        try:
            body = p.read_bytes()
        except FileNotFoundError:
            return self._err(404, "not found")
        except OSError:
            return self._err(500, "cannot read asset")
        self._send(200, body, ctype, extra={"Cache-Control": "no-cache"})

    # ---------- GET ----------

    def _get(self, path: str, qs: dict, rid: str) -> None:
        if not self._limit("general"):
            return

        if path == "/api/health":
            return self._json(self._health())
        if path == "/api/stats":
            return self._json(get_index().stats())
        if path == "/api/config":
            return self._json(CFG.public_dict())
        if path == "/api/suggest":
            raw = (qs.get("q") or [""])[0]
            q, _ = sanitize_text(raw, CFG.max_query_chars)
            lim = self._int_param(qs, "limit", 12, 1, 50)
            return self._json({"suggestions": get_index().suggest(q, limit=lim)})
        if path == "/api/tags":
            cat = (qs.get("category") or [""])[0]
            if cat and cat not in CATEGORIES:
                return self._err(400, f"unknown category: {cat}", rid)
            lim = self._int_param(qs, "limit", 200, 1, 1000)
            idx = get_index()
            tags = idx.by_category(cat, limit=lim) if cat else idx.tags[:lim]
            return self._json({"tags": [t.name for t in tags]})
        if path == "/api/models":
            if not CFG.ai_active:
                return self._json({"models": []})
            if not self._limit("ai"):
                return
            return self._json({"models": ai_mod.list_keyless_models()})
        if path == "/api/related":
            tag = (qs.get("tag") or [""])[0]
            clean, _ = sanitize_text(tag, 200)
            if not clean:
                return self._err(400, "tag is required", rid)
            idx = get_index()
            if not idx.exists(clean):
                return self._err(404, f"unknown tag: {clean}", rid)
            lim = self._int_param(qs, "limit", 10, 1, 50)
            return self._json({
                "tag": clean,
                "related": idx.related(clean, limit=lim),
                "similar": idx.similar(clean, limit=5),
            })
        if path == "/api/interpret":
            raw = (qs.get("q") or [""])[0]
            q, _ = sanitize_text(raw, CFG.max_query_chars)
            if not q:
                return self._err(400, "q is required", rid)
            idx = get_index()
            intent = idx.interpret(q)
            if intent is None:
                return self._json({"available": False})
            return self._json({
                "available": True,
                "energy": intent.energy,
                "wants_vocals": intent.wants_vocals,
                "use_case": intent.use_case,
                "feeling": intent.feeling,
                "categories": intent.categories,
                "guessed_categories": intent.guessed_categories,
                "unknown_words": intent.unknown_words,
                "summary": intent.describe(),
            })
        return self._err(404, "no such route", rid)

    @staticmethod
    def _int_param(qs: dict, name: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int((qs.get(name) or [str(default)])[0])
        except (TypeError, ValueError):
            return default
        return max(lo, min(val, hi))

    # ---------- POST ----------

    def _post(self, path: str, rid: str) -> None:
        if path != "/api/search":
            return self._err(404, "no such route", rid)

        body, err = self._body()
        if err:
            return self._err(400, err, rid)

        # Clean the query: strip control characters, bound the length.
        query, _ = sanitize_text(body.get("query", ""), CFG.max_query_chars)
        if not query:
            return self._err(400, "query is required", rid)

        use_ai = bool(body.get("use_ai", False)) and CFG.ai_active
        provider = str(body.get("provider") or "").strip().lower()
        if provider and provider not in ("keyless", "compat", "off"):
            return self._err(400, "provider must be keyless, compat or off", rid)
        model, _ = sanitize_text(str(body.get("model") or ""), 120)

        limit = self._clamp_int(body.get("limit"), 40, 1, 200)
        timeout = self._clamp_int(body.get("timeout"), CFG.ai_timeout, 3, 120)
        temperature = self._clamp_float(body.get("temperature"), CFG.ai_temperature, 0.0, 2.0)

        # AI requests are expensive: a stricter budget and a real cost.
        if use_ai:
            if not self._limit("ai", cost=1.0):
                return
        else:
            if not self._limit("general"):
                return

        def clean(v, max_items: int = 40) -> list[str]:
            if isinstance(v, str):
                parts = v.split(",")
            elif isinstance(v, list):
                parts = [str(x) for x in v]
            else:
                return []
            out = []
            for p in parts[:max_items]:
                s, _ = sanitize_text(p, 120)
                if s:
                    out.append(s)
            return out

        try:
            outcome = pipeline.search(
                get_index(),
                query=query,
                limit=limit,
                use_ai=use_ai,
                provider=provider or CFG.ai_provider,
                model=model,
                include=clean(body.get("include")),
                exclude=clean(body.get("exclude")),
                categories=clean(body.get("categories"), max_items=5),
                timeout=timeout,
                temperature=temperature,
                cfg=CFG,
                service=get_ai_service() if use_ai else None,
            )
        except Exception as exc:
            sys.stderr.write(f"[{rid}] search failed: {type(exc).__name__}: {exc}\n")
            return self._err(500, safe_error(exc, CFG.debug), rid)

        self._json(outcome.to_dict())

    @staticmethod
    def _clamp_int(v, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(int(v), hi))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_float(v, default: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(float(v), hi))
        except (TypeError, ValueError):
            return default

    # ---------- health ----------

    def _health(self) -> dict:
        idx = get_index()
        st = idx.stats()
        svc = get_ai_service()
        return {
            "status": "ok",
            "version": CFG.public_dict()["version"],
            "tags_loaded": st["total"],
            "categories": st["categories"],
            "vocabulary_terms": st.get("vocabulary_terms", 0),
            "derived_built": st.get("derived_built", False),
            "mode": CFG.mode,
            "offline_capable": True,
            "ai_enabled": CFG.ai_active,
            "ai_provider": CFG.ai_provider if CFG.ai_active else "off",
            "ai": svc.health() if CFG.ai_active else
                  {"ok": False, "models": [], "detail": "ai disabled"},
            "security": {
                "auth_required": CFG.auth_required,
                "rate_limit_per_min": CFG.rate_limit_per_min,
                "ai_rate_limit_per_min": CFG.ai_rate_limit_per_min,
                "tls": bool(CFG.tls_cert),
                "max_query_chars": CFG.max_query_chars,
            },
            "ai_stats": svc.stats() if CFG.ai_active else None,
        }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _banner(idx: TagIndex) -> None:
    st = idx.stats()
    summary = CFG.redacted_summary()
    print("Treblo Tag Explorer")
    print(f"  version     : {CFG.public_dict()['version']}")
    print(f"  mode        : {summary['mode']}")
    print(f"  tags loaded : {len(idx)}  ({CFG.tag_file})")
    print(f"  categories  : {st['categories']}")
    print(f"  vocab terms : {st.get('vocabulary_terms', 0)}"
          f"   derived: {st.get('derived_built')}")
    print(f"  AI          : {summary['ai_provider'] if summary['ai_active'] else 'off (deterministic only)'}")
    if summary["api_key_set"]:
        print(f"  api key     : set ({len(CFG.openai_api_key)} chars, not shown)")
    print(f"  auth        : {summary['auth']}")
    print(f"  rate limit  : {summary['rate_limit']}   AI: {CFG.ai_rate_limit_per_min}/min")
    print(f"  threads     : max {summary['max_threads']} concurrent")
    print(f"  tls         : {summary['tls']}")
    print(f"  listening   : http{'s' if CFG.tls_cert else ''}://{CFG.host}:{CFG.port}")

    # Show which names count as same-origin, so a LAN/domain deployment is not
    # a silent 403 surprise.
    try:
        local = sorted(_local_addresses())
    except Exception:
        local = []
    known = sorted({"localhost", "127.0.0.1", "::1"} | set(local) | set(CFG.allowed_hosts))
    print(f"  known names : {', '.join(known[:6])}"
          f"{' ...' if len(known) > 6 else ''}")

    if CFG.loopback_only and CFG.host not in ("127.0.0.1", "::1", "localhost"):
        print()
        print("  WARNING: no AUTH_TOKEN set and bound to a non-loopback address.")
        print("           Anyone who can reach this port can use it. Set AUTH_TOKEN")
        print("           or set HOST=127.0.0.1 if that is not what you want.")

    if CFG.host in ("0.0.0.0", "::") and not CFG.allowed_hosts:
        print()
        print("  NOTE: bound to all interfaces. Requests arriving by an address this")
        print("        process cannot see (typical inside Docker, where only the")
        print("        container's own IP is visible) will be refused as a foreign")
        print("        origin. If you expose this beyond loopback, list the names")
        print("        clients use:  ALLOWED_HOSTS=192.168.1.50,mybox.local")


def main() -> int:
    try:
        idx = get_index()
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    _banner(idx)

    if CFG.ai_active:
        # Warm the AI service so the first user request is not the one that
        # discovers the endpoint is down.
        try:
            h = get_ai_service().health()
            print(f"  ai probe    : {'ok' if h.get('ok') else 'unavailable'} - {h.get('detail')}")
        except Exception as e:
            print(f"  ai probe    : failed ({type(e).__name__})")

    httpd = BoundedThreadingHTTPServer(
        (CFG.host, CFG.port), Handler, max_threads=CFG.max_threads
    )

    if CFG.tls_cert:
        import ssl

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(CFG.tls_cert, CFG.tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
