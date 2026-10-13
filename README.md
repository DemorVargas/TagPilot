# Treblo Tag Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Turn a plain-English description of what you want to hear into **real tags**
from your Treblo vocabulary.

```
you type:  sad slow piano for a rainy night
you get:   piano, late night lo-fi, piano solo, ambient, sad, slowcore, ...
```

4,136 tags from `data/treblo_tags_v3.txt`, plus a co-occurrence graph, a
classifier and a BM25 index built from that list at startup (~0.3s).

**The engine works fully offline.** The AI is an optional layer you can switch
off entirely.

---

## Install

```bash
./install.sh              # interactive, keyless AI (default)
./install.sh --offline    # no AI, zero network calls
./install.sh --local      # local Ollama AI
./install.sh --byok       # your own API key
./install.sh --uninstall  # remove everything
./install.sh --help       # all flags and env vars
```

Installs Docker automatically if missing (Ubuntu, Debian, Fedora, RHEL, Arch,
openSUSE, Alpine). Open <http://localhost:8000> when it finishes.

### Or without Docker

Standard library only. No pip install, no build step.

```bash
python3 app/server.py
```

---

## Modes

Set with `MODE`:

| `MODE` | What it does | Needs internet? |
|---|---|---|
| `hybrid` (default) | Engine always runs; AI re-rank attempted if reachable | only for AI |
| `offline` | AI disabled at the config level. Zero outbound calls. | **No** |

In offline mode `use_ai` is refused in the request handler, not just hidden in
the UI: the request comes back with `ai.used = false` **and real results**.

### AI options

| Mode | Env | Key? | Internet? |
|---|---|---|---|
| **Keyless** (default) | `AI_PROVIDER=keyless` | No | AI calls only |
| **Local** | `AI_PROVIDER=compat` + local URL | No | No |
| **Own key** | `AI_PROVIDER=compat` + real URL | Yes | Yes |
| **Off** | `MODE=offline` | No | No |

`compat` means anything speaking `/v1/chat/completions`: OpenAI, Ollama, LM
Studio, llama.cpp, vLLM, Groq, Together, OpenRouter.

Public keyless endpoints are shared best-effort infrastructure with no uptime
promise. That is why the engine does the real work and the AI layer is wrapped
in a circuit breaker, a cache and a concurrency cap. If the endpoint dies you
keep working.

---

## How it works

```
your sentence
      |
[1] OFFLINE UNDERSTANDING   energy, use case, vocals, tone     no network
      |
[2] BM25 RETRIEVAL          inverted index, prefix + typo       no network
      |
[3] OFFLINE EXPANSION       use-case tags, co-occurrence graph  no network
      |
[4] OPTIONAL AI RE-RANK     model sees the top 120 hits only    online only
      |
[5] MERGE + LABEL           every tag says where it came from
```

**Ranking is BM25, not hand-tuned magic numbers.** A term in 2 tags counts for
far more than one in 800, and IDF computes that automatically. Scores are
rescaled to a 0-1 confidence so the number on screen means something.

**The vocabulary teaches itself.** No dictionary, no embeddings, no model file:

- **Morphology** - `psychedelic`, `psychedelia`, `psychedelics` reduce to one
  stem, so `psychedelia` finds `psychedelic rock` even though it is not a tag.
- **Co-occurrence graph** - tags sharing informative tokens become neighbours.
- **Centroid classifier** - an unknown word is placed in mood / genre /
  instrument space by comparing it to each category's centroid.
- **Use-case expansion** - "gym" maps to seeds, then the graph reaches further.

**The AI cannot lie to you.** It sees a shortlist of 120 real tags and may only
choose from it. Every tag it returns is re-validated against the vocabulary;
anything invented is discarded and reported as *"Blocked N tag(s) the AI made
up"*. A hallucinating model can produce fewer good picks. It cannot produce a
tag that does not exist.

---

## CLI

Works offline always; `--ai` adds the optional online re-rank.

```bash
python3 app/cli.py "sad slow piano for a rainy night"
python3 app/cli.py "party music no drums"
python3 app/cli.py "music for the gym heavy and fast" --explain
python3 app/cli.py "late night city drive" --ai
python3 app/cli.py --related "dark ambient"
python3 app/cli.py --interpret "chill dinner jazz with no vocals"
python3 app/cli.py --stats
python3 app/cli.py --token
```

Flags: `--include`, `--exclude`, `--category`, `--provider`, `--model`,
`--limit`, `--explain`, `--json`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, mode, AI state, circuit, cache |
| GET | `/api/config` | public runtime config (no secrets) |
| GET | `/api/stats` | vocabulary and graph breakdown |
| GET | `/api/suggest?q=psy` | autocomplete |
| GET | `/api/tags?category=mood` | browse a category |
| GET | `/api/related?tag=dark%20ambient` | learned graph neighbours |
| GET | `/api/interpret?q=...` | offline query understanding |
| GET | `/api/models` | models the keyless service offers |
| POST | `/api/search` | the main endpoint |

```bash
curl -X POST localhost:8000/api/search -H 'Content-Type: application/json' -d '{
  "query": "lonely synths at 3am",
  "use_ai": true,
  "exclude": "rap",
  "limit": 20
}'
```

`source` is `engine`, `ai`, `engine+ai` or `engine+use-case`. `signals` lists
what promoted a tag. `confidence` is 0-1.

---

## Security

- Optional bearer token, compared with `hmac.compare_digest` (constant time).
  Generate one: `python3 app/cli.py --token`. Under 16 chars is refused at boot.
- Published on **127.0.0.1 by default**, not reachable from your LAN. Binding a
  non-loopback address with no token prints a loud warning.
- Per-client token bucket, 60/min burst 15. AI routes get a stricter 12/min.
- Bounded thread pool (24) with a 15s keep-alive idle timeout, so open tabs
  cannot starve the pool. Overload returns 503 with valid JSON, never an empty
  body. Request body capped at 64 KB before being read into memory.
- Outbound: `http`/`https` only, no redirects, ambient proxy vars ignored,
  cloud metadata endpoints refused, responses size-capped.
- Strict CSP and hardening headers on every response. No `unsafe-inline`, no
  `unsafe-eval`, no CDNs.
- Path traversal is structurally impossible: static files come from a fixed
  table in code, no request string ever reaches a filesystem path.
- Internal errors are logged server-side; the client gets a category and a
  request id. Stack traces leak architecture.
- Container: uid 10001, `read_only: true`, `cap_drop: [ALL]`,
  `no-new-privileges`, memory capped at 256 MB, 1 CPU.
- Zero third-party Python packages. Nothing to patch.

**Before you expose it:**

```bash
python3 app/cli.py --token    # generate one
export AUTH_TOKEN="..."       # and set it
# then change the port mapping to "8000:8000" in docker-compose.yml
```

---

## Config

| Env var | Default | Meaning |
|---|---|---|
| `MODE` | `hybrid` | `online` \| `offline` \| `hybrid` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | listen address |
| `TAG_FILE` | `data/treblo_tags_v3.txt` | your tag list, one per line |
| `AUTH_TOKEN` | *(empty)* | bearer token, 16+ chars |
| `RATE_LIMIT_PER_MIN` | `60` | general budget |
| `RATE_LIMIT_BURST` | `15` | burst allowance |
| `AI_RATE_LIMIT_PER_MIN` | `12` | stricter AI budget |
| `MAX_THREADS` | `24` | concurrency cap |
| `MAX_BODY_BYTES` | `65536` | request body cap |
| `MAX_QUERY_CHARS` | `500` | query length cap |
| `TRUST_PROXY` | `0` | honour `X-Forwarded-For` |
| `ALLOWED_ORIGINS` | *(empty)* | comma-separated CORS allow-list |
| `ALLOWED_HOSTS` | *(empty)* | names this instance is reached by |
| `TLS_CERT` / `TLS_KEY` | *(empty)* | enable HTTPS |
| `DEBUG` | `0` | detailed error messages |
| `AI_ENABLED` | `1` | `0` disables AI |
| `AI_PROVIDER` | `keyless` | `keyless` \| `compat` \| `off` |
| `KEYLESS_MODEL` | `openai-fast` | keyless model id |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | for `compat` |
| `OPENAI_API_KEY` | *(empty)* | for `compat` |
| `OPENAI_MODEL` | `llama3.2` | for `compat` |
| `AI_TIMEOUT` | `45` | per-call timeout |
| `AI_CACHE_TTL` | `900` | AI response cache seconds |
| `CIRCUIT_FAIL_THRESHOLD` | `3` | failures before the breaker opens |

Invalid config fails loudly at boot with a specific message, not silently at
request time.

---

## Swapping in your own tag list

Replace `data/treblo_tags_v3.txt` (one tag per line, popularity order: earliest
= most popular, which earns a small ranking boost) and rebuild:

```bash
docker compose up -d --build
```

The graph, classifier and BM25 statistics rebuild from whatever file you give
them.

---

## Tests

```bash
python3 -m unittest discover -s tests -v
```

**176 tests, no pytest needed.** They cover normalization, the stemmer, BM25
math, the index, the graph, the classifier, intent parsing, the palette, the
hallucination guard, the AI cache/breaker/concurrency, the pipeline's
degradation path, and every security control listed above.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Notes

- **Python 3.9+**, standard library only. No Flask, no npm, no build step.
- Tag matching is accent- and case-insensitive: `Forró` and `forro` are the same
  tag. Non-Latin tags (Cyrillic, Greek, Chinese, Devanagari) are preserved and
  searchable.
- 4,136 tags load from 4,160 lines; the difference is 24 groups of pure spelling
  variants (`punk - new wave` / `punk/new wave`) that collapse on purpose.
- Binds IPv4. If a tool cannot connect, use `127.0.0.1` explicitly; `localhost`
  can resolve to `::1` first.
