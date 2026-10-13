"""
Pipeline: free text -> ranked tags.

Order of operations (the order matters):

  1. Offline understanding. Read the sentence into a structured intent using
     only the vocabulary and the structures learned from it. No network.
  2. Deterministic BM25 retrieval over the whole vocabulary. Always runs.
  3. Offline expansion: use-case tags, graph neighbours, intent categories.
     Still no network.
  4. Optional AI re-rank, restricted to the retrieved shortlist. Online only.
  5. Merge, dedupe, label every tag with where it came from and why.

Steps 1-3 are the offline engine and they are complete on their own. Step 4 is
a bonus attempted only when the mode allows it, and its failure is never fatal.
That is what lets one build work both online and offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import ai as ai_mod
import intent as intent_mod
from tags import TagIndex, build_palette, norm_key

RERANK_POOL = 120  # how many deterministic hits the AI is allowed to see


@dataclass
class SearchOutcome:
    query: str
    tags: list[dict] = field(default_factory=list)
    palette: dict[str, list[str]] = field(default_factory=dict)
    related: dict[str, list[dict]] = field(default_factory=dict)

    # offline understanding
    understanding: dict = field(default_factory=dict)
    offline: bool = False

    # AI
    ai_used: bool = False
    ai_provider: str = ""
    ai_model: str = ""
    ai_reason: str = ""
    ai_error: str = ""
    ai_dropped: list[str] = field(default_factory=list)
    ai_cached: bool = False
    ai_latency_ms: int = 0

    deterministic_count: int = 0
    elapsed_ms: int = 0
    mode: str = "hybrid"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tags": self.tags,
            "palette": self.palette,
            "related": self.related,
            "understanding": self.understanding,
            "offline": self.offline,
            "ai": {
                "used": self.ai_used,
                "provider": self.ai_provider,
                "model": self.ai_model,
                "reason": self.ai_reason,
                "error": self.ai_error,
                "dropped_hallucinations": self.ai_dropped,
                "cached": self.ai_cached,
                "latency_ms": self.ai_latency_ms,
            },
            "deterministic_count": self.deterministic_count,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode,
        }


def _boost_rows(rows: list[dict], boosted: list[str], amount: float,
                source: str) -> None:
    """Promote tags already present in `rows`; ignore names that are not."""
    lookup = {norm_key(r["tag"]): r for r in rows}
    for i, name in enumerate(boosted):
        row = lookup.get(norm_key(name))
        if row is None:
            continue
        row["score"] = round(row["score"] + amount - i * (amount * 0.04), 3)
        row.setdefault("signals", [])
        if source not in row["signals"]:
            row["signals"].append(source)
        if source not in row.get("source", "engine"):
            row["source"] = "engine+" + source


def search(
    index: TagIndex,
    query: str,
    limit: int = 40,
    use_ai: bool = False,
    provider: str = "keyless",
    model: str = "",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    categories: list[str] | None = None,
    timeout: int = 45,
    temperature: float = 0.3,
    cfg=None,
    service=None,
) -> SearchOutcome:
    """
    Run the full pipeline.

    `service` is an optional ai.AIService. Passing one in lets the server reuse
    a single instance so its circuit breaker, cache and concurrency cap
    accumulate state across requests instead of resetting every call.
    """
    t0 = time.monotonic()
    include = list(include or [])
    exclude = list(exclude or [])
    categories = list(categories or [])

    mode = getattr(cfg, "mode", "hybrid") if cfg else "hybrid"
    out = SearchOutcome(query=query, mode=mode)

    # Resolve config once, up front. Without this an explicit `provider`
    # argument is silently ignored when cfg is None, and AI never runs.
    if use_ai and cfg is None:
        from config import Config

        cfg = Config.from_env()
        if provider:
            cfg.ai_provider = provider
    if use_ai and cfg is not None and provider and provider != cfg.ai_provider:
        # An explicit provider wins over the ambient config, but never over an
        # offline mode: offline means offline.
        if cfg.mode != "offline":
            cfg.ai_provider = provider

    # ---- 1. offline understanding (no network, no model)
    intent = index.interpret(query) if index.has_derived else None
    if intent is not None:
        out.understanding = {
            "energy": intent.energy,
            "wants_vocals": intent.wants_vocals,
            "use_case": intent.use_case,
            "feeling": intent.feeling,
            "categories": intent.categories,
            "guessed_categories": intent.guessed_categories,
            "unknown_words": intent.unknown_words,
            "summary": intent.describe(),
        }
        # Negative words the intent parser found join the exclusion list.
        for neg in intent.negative:
            if neg not in exclude:
                exclude.append(neg)

    # ---- 2. deterministic BM25 retrieval (always runs)
    base = index.search(
        query,
        limit=max(limit, RERANK_POOL),
        include=include,
        exclude=exclude,
        categories=categories,
    )
    out.deterministic_count = len(base)

    # ---- 3. offline expansion

    # Use-case tags: "for the gym" -> energetic, hard rock, ... via the graph.
    if intent is not None and intent.use_case and index.graph is not None:
        uc = intent_mod.use_case_tags(intent, index, index.graph)
        _boost_rows(base, uc, 2.2, "use-case")

    # Vocals intent is a hard filter, because "no vocals" is a requirement,
    # not a preference. Only applied if it leaves something behind.
    if intent is not None and intent.wants_vocals != "any":
        want_inst = intent.wants_vocals == "no"
        filtered = []
        for r in base:
            t = index.get(r["tag"])
            if t is None:
                filtered.append(r)
                continue
            if want_inst and "vocal" in t.categories:
                continue
            if not want_inst and "instrumental" in t.categories:
                continue
            filtered.append(r)
        if filtered:
            base = filtered

    # Energy intent is a soft bias, not a filter, because "slow" is fuzzy.
    if intent is not None and intent.energy != "neutral":
        for r in base:
            t = index.get(r["tag"])
            if t is None:
                continue
            if intent.energy == "high" and t.norm in intent_mod._ENERGY_HIGH:
                r["score"] = round(r["score"] + 0.9, 3)
            elif intent.energy == "low" and t.norm in intent_mod._ENERGY_LOW:
                r["score"] = round(r["score"] + 0.9, 3)
        base.sort(key=lambda r: (-r["score"], r["tag"]))

    # ---- 4. optional AI re-rank, restricted to the shortlist
    ai_tags: list[str] = []
    ai_exclude: list[str] = []

    offline_mode = (mode == "offline") or not getattr(cfg, "ai_active", False)
    out.offline = offline_mode

    if use_ai and base and not offline_mode:
        pool = [r["tag"] for r in base[:RERANK_POOL]]

        svc = service
        if svc is None:
            from config import Config

            cfg = cfg or Config.from_env()
            svc = ai_mod.AIService(cfg)

        res = svc.interpret(
            index,
            query,
            candidates=pool,
            model=model,
            max_tags=min(limit, 12),
            timeout=timeout,
            temperature=temperature,
            extra_context=(
                f"User already asked to include: {', '.join(include)}. "
                f"User already asked to exclude: {', '.join(exclude)}."
                if (include or exclude)
                else ""
            ),
        )
        out.ai_used = res.ok
        out.ai_provider = res.provider
        out.ai_model = res.model
        out.ai_reason = res.reason
        out.ai_error = res.error
        out.ai_dropped = res.dropped
        out.ai_cached = res.cached
        out.ai_latency_ms = res.latency_ms
        ai_tags = res.tags
        ai_exclude = res.exclude

    # ---- 5. merge
    merged: dict[str, dict] = {}
    for r in base:
        merged[norm_key(r["tag"])] = {
            "tag": r["tag"],
            "score": r["score"],
            "matched_on": r["matched_on"],
            "categories": r["categories"],
            "source": r.get("source", "engine"),
            "signals": r.get("signals", []),
            "ai_rank": None,
        }

    # AI picks get promoted and marked.
    for rank, tag in enumerate(ai_tags):
        key = norm_key(tag)
        if key in merged:
            merged[key]["source"] = "engine+ai"
            merged[key]["ai_rank"] = rank
            merged[key]["score"] = round(merged[key]["score"] + 3.0 - rank * 0.15, 3)
        else:
            t = index.get(tag)
            merged[key] = {
                "tag": tag,
                "score": round(3.5 - rank * 0.15, 3),
                "matched_on": ["ai"],
                "categories": list(t.categories) if t else [],
                "source": "ai",
                "signals": ["ai"],
                "ai_rank": rank,
            }

    # AI-proposed exclusions, unless the user explicitly asked for that tag.
    hard_include = {norm_key(x) for x in include}
    for tag in ai_exclude:
        key = norm_key(tag)
        if key in hard_include:
            continue
        merged.pop(key, None)

    ordered = sorted(merged.values(), key=lambda r: (-r["score"], r["tag"]))

    # Recompute confidence against the final ordering, so the numbers the user
    # sees match the order they see.
    top = ordered[0]["score"] if ordered else 0.0
    if top > 0:
        for r in ordered:
            r["confidence"] = round(r["score"] / top, 4)
    for r in ordered:
        r.setdefault("confidence", 0.0)
        r.setdefault("signals", [])

    out.tags = ordered[:limit]

    # ---- palette + related, both computed offline
    seeds = [r["tag"] for r in out.tags[:4]]
    if seeds:
        out.palette = build_palette(index, seeds)
        if index.has_derived:
            for s in seeds[:3]:
                rel = index.related(s, limit=6)
                if rel:
                    out.related[s] = rel

    out.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return out
