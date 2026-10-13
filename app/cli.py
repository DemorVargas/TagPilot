"""
CLI for the Treblo Tag Explorer.

The CLI is always offline-capable: it works with no network, and --ai simply
adds the optional online re-rank on top.

Examples:
  python3 app/cli.py "sad slow piano for a rainy night"
  python3 app/cli.py "party music no drums"
  python3 app/cli.py "music for the gym heavy and fast" --explain
  python3 app/cli.py "late night city drive" --ai
  python3 app/cli.py "chill lofi" --ai --json
  python3 app/cli.py --related "dark ambient"
  python3 app/cli.py --interpret "chill dinner jazz with no vocals"
  python3 app/cli.py --suggest psy
  python3 app/cli.py --stats
  python3 app/cli.py --health
  python3 app/cli.py --token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import ai as ai_mod  # noqa: E402
import pipeline  # noqa: E402
from config import Config  # noqa: E402
from security import make_token  # noqa: E402
from tags import TagIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Treblo tag explorer (CLI). Works offline; --ai adds an online re-rank."
    )
    ap.add_argument("query", nargs="?", default="", help="what you want to hear")
    ap.add_argument("--ai", action="store_true", help="use AI to re-rank the shortlist")
    ap.add_argument("--provider", default="", help="keyless | compat (default: from env)")
    ap.add_argument("--model", default="", help="model name (blank = provider default)")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--include", default="", help="comma separated must-have tags")
    ap.add_argument("--exclude", default="", help="comma separated never-show tags")
    ap.add_argument("--category", default="", help="mood,genre,instrument,vocal,era")
    ap.add_argument("--explain", action="store_true", help="show score breakdown")
    ap.add_argument("--json", action="store_true", help="raw JSON output")
    ap.add_argument("--suggest", metavar="PREFIX", help="autocomplete tags")
    ap.add_argument("--related", metavar="TAG", help="show learned neighbours of a tag")
    ap.add_argument("--similar", metavar="TAG", help="show near-synonym tags")
    ap.add_argument("--interpret", metavar="QUERY", help="show offline query understanding")
    ap.add_argument("--stats", action="store_true", help="vocabulary stats")
    ap.add_argument("--health", action="store_true", help="check AI reachability")
    ap.add_argument("--token", action="store_true", help="generate an AUTH_TOKEN")
    args = ap.parse_args()

    if args.token:
        print(make_token())
        return 0

    cfg = Config.from_env()
    idx = TagIndex(cfg.tag_file)

    if args.stats:
        st = idx.stats()
        print(f"total tags      : {st['total']}")
        print(f"vocabulary terms: {st.get('vocabulary_terms', 0)}")
        print(f"avg tokens/tag  : {st.get('avg_tokens_per_tag', 0)}")
        if "graph" in st:
            g = st["graph"]
            print(f"graph           : {g['tags_with_neighbours']} tags with neighbours, "
                  f"{g['stems']} stems")
        for k, v in sorted(st["categories"].items(), key=lambda x: -x[1]):
            print(f"  {k:12s} {v}")
        return 0

    if args.suggest:
        for s in idx.suggest(args.suggest, limit=20):
            print(s)
        return 0

    if args.related:
        rel = idx.related(args.related, limit=15)
        if not rel:
            print(f"no graph neighbours for {args.related!r} (is it a tag?)")
            return 1
        print(f"neighbours of {args.related!r} (learned from shared tokens):")
        for r in rel:
            shared = ",".join(r["shared"]) or "-"
            print(f"  {r['tag']:<34} w={r['weight']:<7} shared={shared}")
        return 0

    if args.similar:
        sim = idx.similar(args.similar, limit=10)
        print(f"near-synonyms of {args.similar!r}: {', '.join(sim) if sim else 'none'}")
        return 0

    if args.interpret:
        it = idx.interpret(args.interpret)
        if it is None:
            print("derived structures not built")
            return 1
        print(f'query       : "{args.interpret}"')
        print(f"reading     : {it.describe()}")
        print(f"energy      : {it.energy}")
        print(f"vocals      : {it.wants_vocals}")
        print(f"use case    : {it.use_case or '-'}")
        print(f"terms       : {', '.join(it.positive) or '-'}")
        print(f"excluded    : {', '.join(it.negative) or '-'}")
        if it.guessed_categories:
            print(f"guessed cats: {it.guessed_categories}")
        if it.unknown_words:
            print(f"unrecognised: {', '.join(it.unknown_words)}")
        return 0

    if args.health:
        print(json.dumps(ai_mod.health(args.provider, timeout=15), indent=2))
        return 0

    if not args.query:
        ap.print_help()
        return 1

    def split(v: str) -> list[str]:
        return [x.strip() for x in v.split(",") if x.strip()]

    out = pipeline.search(
        idx,
        query=args.query,
        limit=args.limit,
        use_ai=args.ai,
        provider=args.provider or cfg.ai_provider,
        model=args.model,
        include=split(args.include),
        exclude=split(args.exclude),
        categories=split(args.category),
        cfg=cfg,
    )

    if args.json:
        print(json.dumps(out.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f'\nquery: "{out.query}"')
    print(f"mode: {out.mode}   engine hits: {out.deterministic_count}   ({out.elapsed_ms} ms)")
    if out.understanding:
        print(f"reading: {out.understanding.get('summary', '-')}")
    if args.ai:
        if out.ai_used:
            cached = " (cached)" if out.ai_cached else ""
            print(f"ai: {out.ai_provider}/{out.ai_model}{cached} - "
                  f"{out.ai_reason or 'picked tags'}")
        else:
            print(f"ai: not used ({out.ai_error or 'unavailable'}) -> offline results only")
        if out.ai_dropped:
            print(f"ai invented and got dropped: {', '.join(out.ai_dropped)}")
    print()

    if not out.tags:
        print("  nothing matched. try broader words, or --category mood")
        return 0

    for i, r in enumerate(out.tags, 1):
        src = {"engine": "  ", "ai": "AI", "engine+ai": "AI+"}.get(r["source"], "  ")
        cats = ",".join(r["categories"]) or "-"
        signals = ",".join(r.get("signals", [])) or ""
        extra = f"  [{signals}]" if signals else ""
        print(f"  {i:2d}. [{src}] {r['tag']:<36s} {r['score']:>6.2f} "
              f"conf={r.get('confidence', 0):.2f} ({cats}){extra}")

    if out.palette:
        print("\n  palette:")
        for cat, tags in out.palette.items():
            if tags:
                print(f"    {cat:11s} {', '.join(tags)}")

    if out.related:
        print("\n  related (learned):")
        for seed, rel in out.related.items():
            names = ", ".join(r["tag"] for r in rel[:4])
            print(f"    {seed:22s} -> {names}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
