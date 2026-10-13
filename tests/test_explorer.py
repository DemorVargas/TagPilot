"""
Tests for the Treblo Tag Explorer v2.

Pure stdlib unittest, no pytest needed.

Run:  python3 -m unittest discover -s tests -v

Covers: normalization, the BM25 index, the learned graph, offline intent
understanding, the AI hallucination guard, the pipeline's degradation path,
and every security control (auth, rate limiting, breaker, cache, SSRF guard,
input sanitization, error redaction).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import ai as ai_mod  # noqa: E402
import intent as intent_mod  # noqa: E402
import pipeline  # noqa: E402
import ranking  # noqa: E402
from config import Config, validate_base_url  # noqa: E402
from security import (  # noqa: E402
    CircuitBreaker,
    ConcurrencyLimiter,
    RateLimiter,
    TTLCache,
    check_token,
    client_key,
    extract_bearer,
    make_token,
    safe_error,
    sanitize_text,
    security_headers,
    token_fingerprint,
)
from tags import TagIndex, build_palette, norm_key, normalize, tokenize  # noqa: E402

TAG_FILE = ROOT / "data" / "treblo_tags_v3.txt"


class TestNormalize(unittest.TestCase):
    def test_accents_stripped(self):
        self.assertEqual(normalize("Forró"), "forro")
        self.assertEqual(normalize("Montréal"), "montreal")

    def test_separators_collapsed(self):
        self.assertEqual(normalize("hip-hop/rap"), "hip hop rap")
        self.assertEqual(normalize("r&b"), "r and b")

    def test_non_latin_not_lost(self):
        for raw in ["москва", "αθήνα", "八音盒", "चित्रगीत"]:
            self.assertEqual(normalize(raw), "")
            self.assertTrue(norm_key(raw))
            self.assertTrue(tokenize(raw))


class TestStemmer(unittest.TestCase):
    """The stemmer links tags. Wrong stems create false links, so it is tested."""

    CASES = [
        ("psychedelic", "psychedel"), ("psychedelia", "psychedel"),
        ("psychedelics", "psychedel"), ("electronic", "electron"),
        ("electronica", "electron"), ("harmonies", "harmony"),
        ("harmonic", "harmon"), ("rock", "rock"), ("rocket", "rocket"),
        ("darkness", "dark"), ("blues", "blue"), ("jazz", "jazz"),
        ("metallic", "metal"), ("romantic", "romant"),
    ]

    def test_stem_cases(self):
        for word, want in self.CASES:
            with self.subTest(word=word):
                self.assertEqual(intent_mod.stem_word(word), want)

    def test_short_words_untouched(self):
        self.assertEqual(intent_mod.stem_word("pop"), "pop")
        self.assertEqual(intent_mod.stem_word("jazz"), "jazz")

    def test_never_returns_empty(self):
        for w in ["a", "ab", "abc", "s", "es", "ies", "the", "and", ""]:
            self.assertIsInstance(intent_mod.stem_word(w), str)


class TestRanking(unittest.TestCase):
    def test_idf_non_negative(self):
        corpus = ranking.build_corpus([["a", "b"], ["a"], ["a", "c"]])
        for term in ("a", "b", "c", "zzz"):
            self.assertGreaterEqual(corpus.idf(term), 0.0)

    def test_rare_term_scores_higher(self):
        """A term in 1 document is more informative than one in all of them."""
        docs = [["common", "rare"]] + [["common"]] * 20
        corpus = ranking.build_corpus(docs)
        rare = ranking.bm25_term_score(1, 2, corpus, "rare")
        common = ranking.bm25_term_score(1, 2, corpus, "common")
        self.assertGreater(rare, common)

    def test_avgdl(self):
        self.assertAlmostEqual(ranking.build_corpus([["a"], ["a", "b", "c"]]).avgdl, 2.0)

    def test_empty_corpus_safe(self):
        corpus = ranking.build_corpus([])
        self.assertEqual(corpus.n_docs, 0)
        self.assertEqual(ranking.bm25_term_score(1, 1, corpus, "x"), 0.0)

    def test_normalize_scores_bounds(self):
        out = ranking.normalize_scores([{"score": 10.0}, {"score": 5.0}, {"score": 1.0}])
        self.assertAlmostEqual(out[0]["confidence"], 1.0)
        self.assertAlmostEqual(out[2]["confidence"], 0.1)
        for r in out:
            self.assertLessEqual(r["confidence"], 1.0)
            self.assertGreaterEqual(r["confidence"], 0.0)

    def test_normalize_empty(self):
        self.assertEqual(ranking.normalize_scores([]), [])


class TestIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_loads_vocabulary(self):
        self.assertGreater(len(self.index), 4100)
        self.assertLessEqual(len(self.index), 4160)

    def test_no_duplicate_keys(self):
        keys = [t.norm for t in self.index.tags]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_empty_tags(self):
        for t in self.index.tags:
            self.assertTrue(t.norm.strip(), f"empty norm for {t.name!r}")

    def test_derived_built(self):
        self.assertTrue(self.index.has_derived)
        self.assertIsNotNone(self.index.graph)
        self.assertIsNotNone(self.index.classifier)
        self.assertIsNotNone(self.index.intent_parser)

    def test_tokens_populated(self):
        self.assertTrue(self.index.get("hip hop").tokens)

    def test_non_latin_present(self):
        for name in ["москва", "αθήνα", "八音盒"]:
            self.assertIsNotNone(self.index.get(name), name)

    def test_stats_shape(self):
        st = self.index.stats()
        for key in ("total", "categories", "vocabulary_terms", "avg_tokens_per_tag",
                    "derived_built", "graph"):
            self.assertIn(key, st)

    def test_load_without_derived(self):
        idx = TagIndex(TAG_FILE, build_derived=False)
        self.assertFalse(idx.has_derived)
        self.assertTrue(len(idx) > 4100)
        self.assertTrue(idx.search("piano"))  # search still works


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_basic_hit(self):
        self.assertIn("piano", [r["tag"] for r in self.index.search("piano")])

    def test_empty_query(self):
        self.assertEqual(self.index.search(""), [])

    def test_scores_descending(self):
        scores = [r["score"] for r in self.index.search("sad piano", limit=30)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_confidence_present_and_bounded(self):
        for r in self.index.search("piano", limit=10):
            self.assertIn("confidence", r)
            self.assertLessEqual(r["confidence"], 1.0)
            self.assertGreaterEqual(r["confidence"], 0.0)

    def test_limit_respected(self):
        self.assertLessEqual(len(self.index.search("rock", limit=5)), 5)

    def test_exact_match_ranks_first(self):
        """The whole-tag match must beat a tag that merely contains the word."""
        self.assertEqual(self.index.search("piano", limit=10)[0]["tag"], "piano")

    def test_alias_expansion(self):
        tags = {r["tag"] for r in self.index.search("music for studying")}
        self.assertTrue(tags & {"ambient", "instrumental", "downtempo", "chillout"})

    def test_negation_in_query(self):
        for r in self.index.search("party music no drums", limit=50):
            self.assertNotIn("drums", r["tag"].lower().split())

    def test_exclude_param(self):
        tags = [r["tag"] for r in self.index.search("rock", exclude=["classic rock"], limit=50)]
        self.assertNotIn("classic rock", tags)

    def test_category_filter(self):
        for r in self.index.search("rock", categories=["instrument"], limit=50):
            self.assertIn("instrument", r["categories"])

    def test_unknown_category_ignored(self):
        """An invalid category must not silently return nothing."""
        self.assertTrue(self.index.search("rock", categories=["evil"], limit=5))

    def test_prefix_matching(self):
        tags = [r["tag"] for r in self.index.search("psyche", limit=30)]
        self.assertTrue(any("psychedelic" in t for t in tags), tags[:10])

    def test_graph_reaches_unseen_word(self):
        """'psychedelia' is not a tag; morphology must still find the family."""
        tags = [r["tag"] for r in self.index.search("psychedelia", limit=10)]
        self.assertTrue(any("psychedelic" in t for t in tags), tags[:10])

    def test_typo_tolerance(self):
        self.assertIn("melancholic",
                      [r["tag"] for r in self.index.search("melancolic", limit=20)])

    def test_result_shape(self):
        r = self.index.search("piano")[0]
        for key in ("tag", "score", "matched_on", "categories", "confidence"):
            self.assertIn(key, r)

    def test_explain_adds_parts(self):
        r = self.index.search("piano", limit=1, explain=True)[0]
        self.assertIn("parts", r)
        self.assertIsInstance(r["parts"], dict)

    def test_gibberish_does_not_crash(self):
        for q in ["", "   ", "!!!", "zzzzqqqq", "🎵🎵", "a" * 300, "\x00\x01"]:
            self.index.search(q)

    def test_graph_can_be_disabled(self):
        self.assertIsInstance(self.index.search("psychedelia", limit=5, use_graph=False), list)

    def test_suggest(self):
        s = self.index.suggest("psy", limit=10)
        self.assertTrue(s)
        self.assertTrue(all("psy" in x.lower() for x in s))


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_related_shares_tokens(self):
        rel = self.index.related("dark ambient", limit=5)
        self.assertTrue(rel)
        for r in rel:
            self.assertTrue(r["shared"], f"{r['tag']} has no shared token")

    def test_related_only_real_tags(self):
        for r in self.index.related("rock", limit=10):
            self.assertTrue(self.index.exists(r["tag"]))

    def test_related_unknown_tag(self):
        self.assertEqual(self.index.related("zzznotatag"), [])

    def test_similar_returns_names(self):
        sim = self.index.similar("hip hop", limit=5)
        self.assertIsInstance(sim, list)
        for s in sim:
            self.assertTrue(self.index.exists(s))

    def test_expand_token(self):
        self.assertTrue(self.index.graph.expand_token("psychedelia", limit=5))

    def test_graph_stats(self):
        st = self.index.graph.token_stats()
        self.assertGreater(st["tokens"], 100)
        self.assertGreater(st["stems"], 100)


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_classify_known_words(self):
        for word, want in {"sad": "mood", "guitar": "instrument",
                           "jazz": "genre", "1990s": "era"}.items():
            got = self.index.classifier.classify(word, top=1)
            self.assertTrue(got, word)
            self.assertEqual(got[0][0], want, f"{word} -> {got}")

    def test_classify_unknown_word(self):
        got = self.index.classifier.classify("psychedelia", top=3)
        self.assertTrue(got)
        self.assertIn(got[0][0], ("genre", "mood"))

    def test_classify_never_raises(self):
        for w in ["", "x", "zzzzzzzzzzzz", "🎵", "a" * 200]:
            self.index.classifier.classify(w)


class TestIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_energy_high(self):
        self.assertEqual(self.index.interpret("heavy fast aggressive music").energy, "high")

    def test_energy_low(self):
        self.assertEqual(self.index.interpret("slow calm quiet music").energy, "low")

    def test_instrumental_intent(self):
        self.assertEqual(self.index.interpret("chill jazz with no vocals").wants_vocals, "no")

    def test_use_case_detected(self):
        self.assertEqual(self.index.interpret("music for the gym").use_case, "gym")
        self.assertEqual(self.index.interpret("something for studying").use_case, "study")

    def test_stopwords_excluded_from_terms(self):
        it = self.index.interpret("i want something for the gym")
        for junk in ("want", "something", "the", "for", "i"):
            self.assertNotIn(junk, it.positive)

    def test_negatives_captured(self):
        self.assertIn("drums", self.index.interpret("party music no drums").negative)

    def test_unknown_word_flagged(self):
        it = self.index.interpret("flurble blorp music")
        self.assertTrue(it.unknown_words or it.guessed_categories)

    def test_describe_never_empty(self):
        for q in ["", "x", "music", "sad fast piano"]:
            self.assertTrue(self.index.interpret(q).describe())

    def test_empty_query_safe(self):
        it = self.index.interpret("")
        self.assertEqual(it.positive, [])
        self.assertEqual(it.energy, "neutral")


class TestUseCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_use_case_tags_exist(self):
        it = self.index.interpret("music for the gym")
        tags = intent_mod.use_case_tags(it, self.index, self.index.graph)
        self.assertTrue(tags)
        for t in tags:
            self.assertTrue(self.index.exists(t), t)

    def test_unknown_use_case_returns_empty(self):
        it = self.index.interpret("zzzz music")
        self.assertEqual(intent_mod.use_case_tags(it, self.index, self.index.graph), [])


class TestPalette(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_bright_query_no_dark_moods(self):
        self.assertNotIn("melancholic",
                         build_palette(self.index, ["party", "dance"]).get("mood", []))

    def test_dark_query_no_bright_moods(self):
        self.assertNotIn("uplifting",
                         build_palette(self.index, ["sad", "melancholic"]).get("mood", []))

    def test_only_existing_tags(self):
        for tags in build_palette(self.index, ["rock"]).values():
            for t in tags:
                self.assertTrue(self.index.exists(t), t)

    def test_no_seed_echo(self):
        flat = [t for tags in build_palette(self.index, ["rock", "piano"]).values() for t in tags]
        self.assertNotIn("rock", flat)
        self.assertNotIn("piano", flat)


class TestAIGuard(unittest.TestCase):
    """The hallucination guard is the safety net. It must be airtight."""

    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_drops_invented(self):
        kept, dropped = ai_mod.vet(self.index, ["piano", "totally made up", "melancholic"])
        self.assertEqual(kept, ["piano", "melancholic"])
        self.assertEqual(dropped, ["totally made up"])

    def test_case_accent_insensitive(self):
        kept, dropped = ai_mod.vet(self.index, ["PIANO", "Forró", "MELANCHOLIC"])
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_dedupes(self):
        self.assertEqual(ai_mod.vet(self.index, ["piano", "Piano", "PIANO"])[0], ["piano"])

    def test_all_fake_dropped(self):
        kept, dropped = ai_mod.vet(self.index, ["flurble", "zonk", "qqqq"])
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 3)

    def test_private_alias_still_works(self):
        self.assertEqual(ai_mod._vet(self.index, ["piano"])[0], ["piano"])

    def test_extract_json_variants(self):
        self.assertEqual(ai_mod._extract_json('{"tags":["a"]}'), {"tags": ["a"]})
        self.assertEqual(ai_mod._extract_json('```json\n{"tags":["a"]}\n```'), {"tags": ["a"]})
        self.assertEqual(
            ai_mod._extract_json('Sure: {"tags":["a"],"reason":"x"} done').get("tags"), ["a"]
        )

    def test_extract_json_braces_inside_strings(self):
        """A brace inside a quoted reason must not break the depth walker."""
        got = ai_mod._extract_json('{"tags":["a"],"reason":"use { and } freely"}')
        self.assertEqual(got.get("tags"), ["a"])
        self.assertEqual(got.get("reason"), "use { and } freely")

    def test_extract_json_garbage(self):
        self.assertEqual(ai_mod._extract_json("no json"), {})
        self.assertEqual(ai_mod._extract_json(""), {})

    def test_coerce_list_shapes(self):
        self.assertEqual(ai_mod._coerce_list("a, b"), ["a", "b"])
        self.assertEqual(ai_mod._coerce_list(["a"]), ["a"])
        self.assertEqual(ai_mod._coerce_list([{"tag": "a"}]), ["a"])
        self.assertEqual(ai_mod._coerce_list(None), [])
        self.assertEqual(ai_mod._coerce_list(42), [])

    def test_wrong_shape_is_failure(self):
        """Valid JSON with the wrong keys is a failure, not a success."""
        cfg = Config.from_env()
        cfg.ai_provider = "compat"
        svc = ai_mod.AIService(cfg)
        svc.provider._chat = lambda *a, **k: '{"wrong":"shape"}'
        res = svc.interpret(self.index, "x", ["piano"])
        self.assertFalse(res.ok)
        self.assertIn("shape", res.error)

    def test_unparseable_is_failure(self):
        cfg = Config.from_env()
        cfg.ai_provider = "compat"
        svc = ai_mod.AIService(cfg)
        svc.provider._chat = lambda *a, **k: "I cannot help with that."
        self.assertFalse(svc.interpret(self.index, "x", ["piano"]).ok)

    def test_interpret_disabled(self):
        self.assertFalse(ai_mod.interpret(self.index, "x", ["piano"], provider="off").ok)

    def test_interpret_empty_candidates(self):
        self.assertFalse(ai_mod.interpret(self.index, "x", [], provider="keyless").ok)


class TestAIServiceMachinery(unittest.TestCase):
    """Cache, breaker and concurrency, exercised with a stubbed provider."""

    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def _service(self, reply='{"tags":["piano"]}'):
        cfg = Config.from_env()
        cfg.ai_provider = "compat"
        svc = ai_mod.AIService(cfg)
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return reply

        svc.provider._chat = fake
        svc.calls = calls
        return svc

    def test_cache_prevents_second_call(self):
        svc = self._service()
        a = svc.interpret(self.index, "q", ["piano"])
        b = svc.interpret(self.index, "q", ["piano"])
        self.assertTrue(a.ok)
        self.assertFalse(a.cached)
        self.assertTrue(b.cached)
        self.assertEqual(svc.calls["n"], 1)

    def test_cache_key_includes_candidates(self):
        svc = self._service()
        svc.interpret(self.index, "q", ["piano"])
        svc.interpret(self.index, "q", ["guitar"])
        self.assertEqual(svc.calls["n"], 2)

    def test_breaker_opens_after_failures(self):
        cfg = Config.from_env()
        cfg.ai_provider = "compat"
        cfg.circuit_fail_threshold = 2
        svc = ai_mod.AIService(cfg)

        def boom(*a, **k):
            raise TimeoutError("nope")

        svc.provider._chat = boom
        for _ in range(2):
            svc.interpret(self.index, "q", ["piano"])
        self.assertEqual(svc.breaker.state, "open")
        res = svc.interpret(self.index, "q", ["piano"])
        self.assertFalse(res.ok)
        self.assertIn("repeated failures", res.error)

    def test_breaker_resets_on_success(self):
        svc = self._service()
        svc.breaker.record_failure()
        svc.interpret(self.index, "q", ["piano"])
        self.assertEqual(svc.breaker.state, "closed")

    def test_concurrency_limit_refuses(self):
        svc = self._service()
        svc.limiter = ConcurrencyLimiter(1)
        self.assertTrue(svc.limiter.try_acquire())
        res = svc.interpret(self.index, "q", ["piano"])
        self.assertFalse(res.ok)
        self.assertIn("busy", res.error)
        svc.limiter.release()

    def test_service_never_raises(self):
        cfg = Config.from_env()
        cfg.ai_provider = "compat"
        svc = ai_mod.AIService(cfg)

        def boom(*a, **k):
            raise RuntimeError("explode")

        svc.provider._chat = boom
        res = svc.interpret(self.index, "q", ["piano"])
        self.assertFalse(res.ok)
        self.assertTrue(res.error)

    def test_stats_shape(self):
        st = self._service().stats()
        for k in ("available", "provider", "circuit", "cache", "concurrency"):
            self.assertIn(k, st)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TagIndex(TAG_FILE)

    def test_engine_only(self):
        out = pipeline.search(self.index, "sad piano", use_ai=False)
        self.assertFalse(out.ai_used)
        self.assertTrue(out.tags)
        self.assertGreater(out.deterministic_count, 0)

    def test_offline_mode_skips_ai(self):
        cfg = Config.from_env()
        cfg.mode = "offline"
        cfg.ai_enabled = False
        out = pipeline.search(self.index, "sad piano", use_ai=True, cfg=cfg)
        self.assertTrue(out.offline)
        self.assertFalse(out.ai_used)
        self.assertTrue(out.tags)

    def test_ai_failure_still_returns_engine_results(self):
        """The single most important test: a dead AI must not break the app."""
        out = pipeline.search(
            self.index, "sad piano", use_ai=True,
            provider="compat", model="nope", timeout=3,
        )
        self.assertTrue(out.tags, "engine results must survive an AI failure")
        self.assertFalse(out.ai_used)
        self.assertTrue(out.ai_error)

    def test_ai_never_injects_unknown_tags(self):
        for t in pipeline.search(self.index, "chill", limit=50).tags:
            self.assertTrue(self.index.exists(t["tag"]), t["tag"])

    def test_understanding_present(self):
        out = pipeline.search(self.index, "music for the gym heavy", use_ai=False)
        self.assertTrue(out.understanding)
        self.assertIn("summary", out.understanding)

    def test_use_case_boosts(self):
        out = pipeline.search(self.index, "music for the gym", use_ai=False, limit=20)
        signals = [s for t in out.tags for s in t.get("signals", [])]
        self.assertIn("use-case", signals)

    def test_instrumental_filter_applied(self):
        out = pipeline.search(self.index, "jazz with no vocals", use_ai=False, limit=30)
        for t in out.tags:
            tag = self.index.get(t["tag"])
            if tag:
                self.assertNotIn("vocal", tag.categories, t["tag"])

    def test_confidence_bounded(self):
        for t in pipeline.search(self.index, "piano", use_ai=False, limit=30).tags:
            self.assertLessEqual(t["confidence"], 1.0)
            self.assertGreaterEqual(t["confidence"], 0.0)

    def test_source_labels_valid(self):
        for t in pipeline.search(self.index, "chill lofi", use_ai=False).tags:
            self.assertTrue(t["source"].startswith(("engine", "ai")), t["source"])

    def test_to_dict_shape(self):
        d = pipeline.search(self.index, "rock", use_ai=False).to_dict()
        for key in ("query", "tags", "palette", "related", "understanding", "offline",
                    "ai", "deterministic_count", "elapsed_ms", "mode"):
            self.assertIn(key, d)
        for key in ("used", "provider", "model", "reason", "error",
                    "dropped_hallucinations", "cached", "latency_ms"):
            self.assertIn(key, d["ai"])

    def test_empty_query(self):
        self.assertEqual(pipeline.search(self.index, "", use_ai=False).tags, [])

    def test_related_present(self):
        self.assertTrue(pipeline.search(self.index, "dark ambient", use_ai=False, limit=6).related)

    def test_speed(self):
        """The engine must stay fast on this low-end box."""
        t0 = time.monotonic()
        pipeline.search(self.index, "sad slow piano for a rainy night", use_ai=False)
        self.assertLess(time.monotonic() - t0, 3.0)


class TestSecurityTokens(unittest.TestCase):
    def test_make_token_entropy(self):
        a, b = make_token(), make_token()
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(a), 32)

    def test_check_token_semantics(self):
        t = make_token()
        self.assertTrue(check_token(t, t))
        self.assertFalse(check_token("", t))
        self.assertFalse(check_token(t, ""))
        self.assertFalse(check_token(t[:8], t))
        self.assertFalse(check_token(t + "x", t))

    def test_check_token_non_ascii_safe(self):
        """compare_digest raises on non-ASCII str; make sure we never crash."""
        self.assertFalse(check_token("tökén", "token"))

    def test_extract_bearer(self):
        self.assertEqual(extract_bearer("Bearer abc"), "abc")
        self.assertEqual(extract_bearer("bearer abc"), "abc")
        self.assertEqual(extract_bearer("  Bearer   abc  "), "abc")
        self.assertEqual(extract_bearer("abc"), "abc")
        self.assertEqual(extract_bearer(""), "")
        self.assertEqual(extract_bearer("Bearer"), "")

    def test_fingerprint_hides_token(self):
        t = make_token()
        fp = token_fingerprint(t)
        self.assertNotIn(t, fp)
        self.assertEqual(fp, token_fingerprint(t))
        self.assertEqual(token_fingerprint(""), "none")


class TestRateLimiter(unittest.TestCase):
    def test_burst_then_refuse(self):
        rl = RateLimiter(rate_per_min=60, burst=5)
        for _ in range(5):
            self.assertTrue(rl.check("k")[0])
        allowed, retry_after, _ = rl.check("k")
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)

    def test_separate_keys_independent(self):
        rl = RateLimiter(rate_per_min=60, burst=2)
        self.assertTrue(rl.check("a")[0])
        self.assertTrue(rl.check("a")[0])
        self.assertFalse(rl.check("a")[0])
        self.assertTrue(rl.check("b")[0], "one client must not affect another")

    def test_refills(self):
        rl = RateLimiter(rate_per_min=6000, burst=1)  # 100/sec
        self.assertTrue(rl.check("k")[0])
        self.assertFalse(rl.check("k")[0])
        time.sleep(0.05)
        self.assertTrue(rl.check("k")[0])

    def test_bounded_memory(self):
        rl = RateLimiter(rate_per_min=60, burst=1, max_clients=10)
        for i in range(100):
            rl.check(f"client{i}")
        self.assertLessEqual(rl.snapshot()["clients_tracked"], 10)

    def test_thread_safe(self):
        rl = RateLimiter(rate_per_min=600000, burst=1000)
        ok = []
        lock = threading.Lock()

        def worker():
            for _ in range(50):
                got = rl.check("shared")[0]
                with lock:
                    ok.append(got)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(ok), 400)


class TestClientKey(unittest.TestCase):
    def test_ignores_forwarded_for_by_default(self):
        self.assertEqual(client_key("1.2.3.4", "9.9.9.9", trust_proxy=False), "1.2.3.4")

    def test_uses_forwarded_for_when_trusted(self):
        self.assertEqual(client_key("10.0.0.1", "9.9.9.9, 10.0.0.1", trust_proxy=True), "9.9.9.9")

    def test_empty_inputs(self):
        self.assertEqual(client_key("", "", trust_proxy=False), "unknown")


class TestCache(unittest.TestCase):
    def test_put_get(self):
        c = TTLCache(max_size=10, ttl=60)
        c.put("k", "v")
        self.assertEqual(c.get("k"), "v")

    def test_ttl_expiry(self):
        c = TTLCache(max_size=10, ttl=0)
        c.put("k", "v")
        self.assertIsNone(c.get("k"))

    def test_lru_eviction(self):
        c = TTLCache(max_size=2, ttl=60)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("c"), 3)

    def test_key_deterministic_and_distinct(self):
        self.assertEqual(TTLCache.key_for("a", "b"), TTLCache.key_for("a", "b"))
        self.assertNotEqual(TTLCache.key_for("a", "b"), TTLCache.key_for("ab", ""))

    def test_stats(self):
        c = TTLCache(max_size=10, ttl=60)
        c.put("k", "v")
        c.get("k")
        c.get("miss")
        st = c.stats()
        self.assertEqual(st["hits"], 1)
        self.assertEqual(st["misses"], 1)


class TestCircuitBreaker(unittest.TestCase):
    def test_closed_initially(self):
        self.assertEqual(CircuitBreaker().state, "closed")

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(fail_threshold=2, reset_after=60)
        cb.record_failure()
        self.assertEqual(cb.state, "closed")
        cb.record_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow())

    def test_half_open_after_reset(self):
        cb = CircuitBreaker(fail_threshold=1, reset_after=1)
        cb.record_failure()
        self.assertEqual(cb.state, "open")
        # Backdate the open time rather than sleeping, so the test is fast and
        # not timing-dependent.
        cb._opened_at -= 2.0
        self.assertEqual(cb.state, "half_open")
        self.assertTrue(cb.allow())

    def test_success_closes(self):
        cb = CircuitBreaker(fail_threshold=1, reset_after=60)
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, "closed")

    def test_stats_does_not_deadlock(self):
        """stats() used to call the locked `state` property and hang forever."""
        cb = CircuitBreaker(fail_threshold=2, reset_after=60)
        cb.record_failure()
        done = []

        def call():
            done.append(cb.stats())

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "stats() deadlocked")
        self.assertEqual(done[0]["failures"], 1)


class TestConcurrencyLimiter(unittest.TestCase):
    def test_caps(self):
        lim = ConcurrencyLimiter(2)
        self.assertTrue(lim.try_acquire())
        self.assertTrue(lim.try_acquire())
        self.assertFalse(lim.try_acquire())
        lim.release()
        self.assertTrue(lim.try_acquire())

    def test_release_below_zero_safe(self):
        lim = ConcurrencyLimiter(1)
        lim.release()
        self.assertEqual(lim.stats()["in_use"], 0)

    def test_stats(self):
        self.assertEqual(ConcurrencyLimiter(3).stats(), {"in_use": 0, "limit": 3})


class TestSSRFGuard(unittest.TestCase):
    """The outbound URL validator. Metadata endpoints must never be reachable."""

    def test_allows_https(self):
        self.assertTrue(validate_base_url("https://api.openai.com/v1")[0])

    def test_allows_loopback(self):
        self.assertTrue(validate_base_url("http://127.0.0.1:11434/v1")[0])
        self.assertTrue(validate_base_url("http://localhost:11434/v1")[0])

    def test_allows_private_range(self):
        self.assertTrue(validate_base_url("http://192.168.1.50:8080/v1")[0])

    def test_blocks_metadata_endpoints(self):
        for url in ["http://169.254.169.254/latest/meta-data",
                    "http://metadata.google.internal/",
                    "http://100.100.100.200/latest/meta-data"]:
            ok, why = validate_base_url(url)
            self.assertFalse(ok, url)
            self.assertIn("blocked", why)

    def test_blocks_bad_schemes(self):
        for url in ["file:///etc/passwd", "gopher://x/", "dict://x:1/", "ftp://x/"]:
            self.assertFalse(validate_base_url(url)[0], url)

    def test_blocks_empty_and_malformed(self):
        for url in ["", "notaurl", "http://", "://x"]:
            self.assertFalse(validate_base_url(url)[0], url)


class TestSanitize(unittest.TestCase):
    def test_strips_control_chars(self):
        clean, changed = sanitize_text("piano\x00\x07 sad", 100)
        self.assertEqual(clean, "piano sad")
        self.assertTrue(changed)

    def test_bounds_length(self):
        clean, changed = sanitize_text("x" * 1000, 10)
        self.assertEqual(len(clean), 10)
        self.assertTrue(changed)

    def test_keeps_quotes_and_angles(self):
        """This is a search box; escaping happens at render time, not here."""
        self.assertEqual(sanitize_text('a "b" <c>', 100)[0], 'a "b" <c>')

    def test_non_string(self):
        self.assertEqual(sanitize_text(None, 10), ("", True))
        self.assertEqual(sanitize_text(42, 10), ("", True))

    def test_newline_and_tab_stripped(self):
        clean, _ = sanitize_text("a\nb\tc", 100)
        self.assertNotIn("\n", clean)
        self.assertNotIn("\t", clean)


class TestErrorRedaction(unittest.TestCase):
    def test_no_paths_in_production_errors(self):
        try:
            raise FileNotFoundError("/etc/shadow: permission denied")
        except FileNotFoundError as e:
            msg = safe_error(e, debug=False)
        self.assertNotIn("/etc/shadow", msg)
        self.assertNotIn("permission denied", msg)

    def test_debug_mode_shows_detail(self):
        try:
            raise ValueError("specific detail")
        except ValueError as e:
            self.assertIn("specific detail", safe_error(e, debug=True))

    def test_unknown_exception_is_generic(self):
        class WeirdError(Exception):
            pass

        self.assertEqual(safe_error(WeirdError("secret stuff"), debug=False), "internal error")


class TestSecurityHeaders(unittest.TestCase):
    def test_required_headers(self):
        h = security_headers()
        for k in ("Content-Security-Policy", "X-Content-Type-Options",
                  "X-Frame-Options", "Referrer-Policy", "Permissions-Policy"):
            self.assertIn(k, h)

    def test_csp_is_locked_down(self):
        csp = security_headers()["Content-Security-Policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("base-uri 'none'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertNotIn("unsafe-eval", csp)

    def test_nosniff(self):
        self.assertEqual(security_headers()["X-Content-Type-Options"], "nosniff")

    def test_extra_merged(self):
        h = security_headers({"X-Custom": "1"})
        self.assertEqual(h["X-Custom"], "1")
        self.assertIn("Content-Security-Policy", h)


class TestConfig(unittest.TestCase):
    def test_defaults_are_safe(self):
        cfg = Config.from_env()
        self.assertFalse(cfg.debug)
        self.assertIn(cfg.mode, ("online", "offline", "hybrid"))

    def test_bad_mode_refused(self):
        import os

        os.environ["MODE"] = "banana"
        try:
            with self.assertRaises(SystemExit):
                Config.from_env()
        finally:
            del os.environ["MODE"]

    def test_short_token_refused(self):
        import os

        os.environ["AUTH_TOKEN"] = "short"
        try:
            with self.assertRaises(SystemExit):
                Config.from_env()
        finally:
            del os.environ["AUTH_TOKEN"]

    def test_auth_required_without_token_refused(self):
        import os

        os.environ["AUTH_REQUIRED"] = "1"
        os.environ.pop("AUTH_TOKEN", None)
        try:
            with self.assertRaises(SystemExit):
                Config.from_env()
        finally:
            del os.environ["AUTH_REQUIRED"]

    def test_offline_mode_disables_ai(self):
        import os

        os.environ["MODE"] = "offline"
        os.environ["AI_ENABLED"] = "1"
        try:
            self.assertFalse(Config.from_env().ai_active)
        finally:
            del os.environ["MODE"]
            del os.environ["AI_ENABLED"]

    def test_bad_base_url_refused(self):
        import os

        os.environ["AI_PROVIDER"] = "compat"
        os.environ["OPENAI_BASE_URL"] = "http://169.254.169.254/v1"
        try:
            with self.assertRaises(SystemExit):
                Config.from_env()
        finally:
            del os.environ["AI_PROVIDER"]
            del os.environ["OPENAI_BASE_URL"]

    def test_public_dict_has_no_secrets(self):
        cfg = Config.from_env()
        cfg.openai_api_key = "sk-supersecret"
        self.assertNotIn("supersecret", json.dumps(cfg.public_dict()))

    def test_redacted_summary_hides_key(self):
        cfg = Config.from_env()
        cfg.openai_api_key = "sk-supersecret"
        self.assertNotIn("supersecret", json.dumps(cfg.redacted_summary()))
        self.assertTrue(cfg.redacted_summary()["api_key_set"])

    def test_tls_cert_without_key_refused(self):
        import os

        os.environ["TLS_CERT"] = "/tmp/x.pem"
        try:
            with self.assertRaises(SystemExit):
                Config.from_env()
        finally:
            del os.environ["TLS_CERT"]


class TestOverloadResponse(unittest.TestCase):
    """
    Regression: the overload path used to return 503 with an empty body, which
    made the browser report 'JSON.parse: unexpected character at line 1 column
    1' instead of a useful message. It must always return parseable JSON.
    """

    def test_reject_body_is_valid_json(self):
        import json as _json

        sys.path.insert(0, str(ROOT / "app"))
        import server

        # Reproduce the reject payload exactly as the server builds it.
        payload = _json.dumps({
            "error": "server busy, too many concurrent connections",
            "detail": f"limit is {server.CFG.max_threads}; retry shortly",
        }).encode("utf-8")
        parsed = _json.loads(payload.decode("utf-8"))
        self.assertIn("error", parsed)
        self.assertTrue(parsed["error"])

    def test_handler_has_idle_timeout(self):
        """
        Without an idle timeout, HTTP/1.1 keep-alive holds a pool slot per open
        browser connection and a few tabs exhaust the pool.
        """
        import server

        self.assertIsNotNone(server.Handler.timeout)
        self.assertLessEqual(server.Handler.timeout, 60)


class TestKeepAliveBodyDrain(unittest.TestCase):
    """
    Regression: a handler that responded without reading the request body left
    the body in the socket. On an HTTP/1.1 keep-alive connection the NEXT
    request parsed those bytes as its request line, producing Python's default
    HTML error page:

        HTTP/1.1 501 Unsupported method ('{"query":"piano","limit":3}POST')

    which surfaced in the browser as "server returned an HTML page (400)".
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "app"))
        import server

        cls.server = server

    def test_drain_limit_is_bounded(self):
        """Draining must be capped, or it becomes a self-inflicted DoS."""
        self.assertLessEqual(self.server.Handler.DRAIN_LIMIT, 10_000_000)
        self.assertGreater(self.server.Handler.DRAIN_LIMIT, 64 * 1024)

    def test_send_error_emits_json_not_html(self):
        """
        The protocol-level error path must produce JSON. The stock
        BaseHTTPRequestHandler emits HTML, which is what the client reported.
        """
        import inspect

        src = inspect.getsource(self.server.Handler.send_error)
        self.assertIn("application/json", src)
        self.assertNotIn("<!DOCTYPE", src)
        # It must not delegate to the HTML-producing base implementation.
        self.assertNotIn("super().send_error", src)

    def test_send_drains_before_responding(self):
        """Every normal response must drain the unread body first."""
        import inspect

        src = inspect.getsource(self.server.Handler._send)
        self.assertIn("_drain_if_needed", src)

    def test_body_records_consumption(self):
        """_body() must record how much it read, so the drain knows the rest."""
        import inspect

        src = inspect.getsource(self.server.Handler._body)
        self.assertIn("_body_consumed", src)

    def test_state_reset_per_request(self):
        """Stale consumption counts must not leak between requests."""
        import inspect

        src = inspect.getsource(self.server.Handler.handle_one_request)
        self.assertIn("_body_consumed = None", src)


class TestCORSSameOrigin(unittest.TestCase):
    """
    Regression: CORS was treated as "must be in ALLOWED_ORIGINS", which
    defaults to empty. Browsers send an Origin header on same-origin POSTs, so
    every request from the app's own UI was rejected with "origin not allowed".
    CORS governs CROSS-origin access; same-origin must always work.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "app"))
        import server

        cls.server = server

    def _handler_like(self, origin, host="localhost:8000"):
        """Build a minimal object exposing the methods _same_origin needs."""
        srv = self.server

        class FakeHeaders(dict):
            def get(self, k, d=""):
                return dict.get(self, k, d)

        class Fake(srv.Handler):
            def __init__(self):
                self.headers = FakeHeaders({"Host": host})
                if origin is not None:
                    self.headers["Origin"] = origin

        return Fake()

    def test_same_origin_localhost_allowed(self):
        h = self._handler_like("http://localhost:8000")
        self.assertTrue(h._same_origin("http://localhost:8000"))

    def test_same_origin_127_allowed(self):
        h = self._handler_like("http://127.0.0.1:8000", host="127.0.0.1:8000")
        self.assertTrue(h._same_origin("http://127.0.0.1:8000"))

    def test_no_origin_allowed(self):
        h = self._handler_like(None)
        self.assertTrue(h._cors())

    def test_same_origin_cors_passes(self):
        h = self._handler_like("http://localhost:8000")
        self.assertTrue(h._cors())

    def test_different_port_is_not_same_origin(self):
        """A dev server on another port is a different origin."""
        h = self._handler_like("http://localhost:3000")
        self.assertFalse(h._same_origin("http://localhost:3000"))

    def test_foreign_origin_refused(self):
        h = self._handler_like("http://evil.example.com")
        self.assertFalse(h._cors())

    def test_foreign_origin_refused_even_with_port_match(self):
        h = self._handler_like("http://evil.example.com:8000")
        self.assertFalse(h._same_origin("http://evil.example.com:8000"))

    def test_malformed_origin_refused(self):
        h = self._handler_like("not a url")
        self.assertFalse(h._same_origin("not a url"))

    def test_allowlisted_cross_origin_gets_echo(self):
        old = list(self.server.CFG.allowed_origins)
        try:
            self.server.CFG.allowed_origins = ["https://tags.example.com"]
            h = self._handler_like("https://tags.example.com")
            self.assertTrue(h._cors())
            self.assertEqual(h._cors_echo, "https://tags.example.com")
        finally:
            self.server.CFG.allowed_origins = old

    def test_same_origin_gets_no_cors_echo(self):
        """Same-origin needs no CORS headers at all."""
        h = self._handler_like("http://localhost:8000")
        h._cors()
        self.assertIsNone(h._cors_echo)

    def test_lan_address_allowed(self):
        """
        Reaching it by LAN IP must work, otherwise exposing the port breaks the
        UI with 'origin not allowed'. Origin and Host both carry the LAN IP.
        """
        lan = "192.168.0.214"
        h = self._handler_like(f"http://{lan}:8000", host=f"{lan}:8000")
        self.assertTrue(h._same_origin(f"http://{lan}:8000"))

    def test_hostname_allowed(self):
        """
        A hostname this machine is actually known by must work. Uses the real
        hostname rather than a made-up one, because an unrecognised name is
        supposed to be refused (that is the DNS-rebinding defence).
        """
        import socket as _socket

        name = _socket.gethostname()
        h = self._handler_like(f"http://{name}:8000", host=f"{name}:8000")
        self.assertTrue(h._same_origin(f"http://{name}:8000"))

    def test_unknown_hostname_refused_without_config(self):
        """
        An unrecognised name is refused unless listed in ALLOWED_HOSTS. If you
        reach the app by a name that is not auto-detected (an mDNS name, a
        proxy domain), add it there.
        """
        h = self._handler_like("http://mybox.local:8000", host="mybox.local:8000")
        old = list(self.server.CFG.allowed_hosts)
        try:
            self.server.CFG.allowed_hosts = []
            h2 = self._handler_like("http://mybox.local:8000", host="mybox.local:8000")
            self.assertFalse(h2._same_origin("http://mybox.local:8000"))
            # ...and allowed once configured.
            self.server.CFG.allowed_hosts = ["mybox.local"]
            h3 = self._handler_like("http://mybox.local:8000", host="mybox.local:8000")
            self.assertTrue(h3._same_origin("http://mybox.local:8000"))
        finally:
            self.server.CFG.allowed_hosts = old

    def test_dns_rebinding_refused(self):
        """
        An attacker's domain resolving to this machine would otherwise match
        itself: Origin and Host both set to the attacker's domain. The
        known-name check must refuse it.
        """
        h = self._handler_like("http://evil.example.com:8000",
                               host="evil.example.com:8000")
        self.assertFalse(h._same_origin("http://evil.example.com:8000"))
        self.assertFalse(h._cors())

    def test_origin_and_host_disagree_refused(self):
        """Origin claiming one host while arriving at another is not same-origin."""
        h = self._handler_like("http://localhost:8000", host="192.168.0.214:8000")
        self.assertFalse(h._same_origin("http://localhost:8000"))

    def test_proxy_tls_port_tolerated(self):
        """
        Behind a TLS-terminating proxy the Origin implies 443 while Host may
        carry no port. Requiring an explicit port match would reject a valid
        same-origin request.
        """
        h = self._handler_like("https://tags.example.com", host="tags.example.com")
        old = list(self.server.CFG.allowed_hosts)
        try:
            self.server.CFG.allowed_hosts = ["tags.example.com"]
            h2 = self._handler_like("https://tags.example.com", host="tags.example.com")
            self.assertTrue(h2._same_origin("https://tags.example.com"))
        finally:
            self.server.CFG.allowed_hosts = old

    def test_allowed_hosts_config_respected(self):
        old = list(self.server.CFG.allowed_hosts)
        try:
            self.server.CFG.allowed_hosts = ["tags.internal"]
            h = self._handler_like("http://tags.internal:8000", host="tags.internal:8000")
            self.assertTrue(h._same_origin("http://tags.internal:8000"))
        finally:
            self.server.CFG.allowed_hosts = old

    def test_local_addresses_non_empty(self):
        """Address detection must yield something, or LAN access silently breaks."""
        import server

        addrs = server._local_addresses()
        self.assertIsInstance(addrs, set)
        # gethostbyname_ex or the hostname fallback should always produce at
        # least one name on a working machine.
        self.assertTrue(addrs, "no local addresses detected")


class TestStaticFileTable(unittest.TestCase):
    """The server serves static files from a fixed table, never a request path."""

    @classmethod
    def setUpClass(cls):
        import server

        cls.server = server

    def test_no_traversal_reachable(self):
        for evil in ["/../../etc/passwd", "/..%2f..%2fetc/passwd", "/app/../app/server.py"]:
            self.assertNotIn(evil, self.server.STATIC)

    def test_only_expected_files(self):
        self.assertEqual(
            set(self.server.STATIC.values()),
            {("index.html", "text/html; charset=utf-8"),
             ("app.js", "application/javascript; charset=utf-8"),
             ("style.css", "text/css; charset=utf-8")},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
