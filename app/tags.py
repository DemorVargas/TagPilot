"""
Tag index: loads the Treblo tag vocabulary and provides fast search.

Pure stdlib. No AI here. This is the deterministic engine that works even if
every network call fails, so the app never becomes useless.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------
# Tag categories. Used for boosting + for the UI to group results.
# A tag can belong to several. Matching is done on normalized text.
# --------------------------------------------------------------------------

MOOD = {
    "sad", "melancholic", "happy", "angry", "dark", "calm", "peaceful", "epic",
    "romantic", "relaxation", "energetic", "rhythmic", "atmospheric", "mellow",
    "warm", "cold", "aggressive", "soft", "party", "summer", "winter", "love",
    "hopeful", "dreamy", "chill", "chillout", "cozy", "hypnotic", "euphoric",
    "nostalgic", "sentimental", "uplifting", "brooding", "somber", "gentle",
    "intense", "playful", "triumphant", "moody", "bittersweet", "soothing",
    "meditative", "spiritual", "sensual", "groovy", "funky", "driving",
    "anthemic", "cinematic", "spacey", "trippy", "ethereal", "haunting",
    "reflective", "introspective", "joyful", "sorrowful", "anxious", "tense",
    "menacing", "sleazy", "smooth", "raw", "gritty", "sweet", "cheerful",
    "lonely", "romantic", "erotic", "ecstatic", "tranquil", "serene",
}

GENRE = {
    "pop", "rock", "rap", "hip hop", "hip-hop/rap", "hip hop/rap", "jazz",
    "blues", "folk", "country", "classical", "electronic", "metal", "punk",
    "soul", "funk", "reggae", "latin", "dance", "house", "techno", "trap",
    "ambient", "disco", "r&b", "r&b/soul", "gospel", "world", "ska", "grunge",
    "dubstep", "trance", "garage", "drum and bass", "dnb", "industrial",
    "bluegrass", "opera", "swing", "bossa nova", "salsa", "cumbia", "afrobeat",
    "k-pop", "j-pop", "emo", "hardcore", "shoegaze", "new age", "easy listening",
    "soundtrack", "original soundtrack", "stage & screen", "spoken word",
    "children", "experimental", "alternative", "indie", "pop rock", "art rock",
}

INSTRUMENT = {
    "guitar", "piano", "bass", "drums", "drums (drum set)", "keyboard",
    "percussion", "violin", "cello", "saxophone", "sax", "trumpet", "flute",
    "organ", "synth", "synthesizer", "strings", "brass", "harp", "banjo",
    "mandolin", "accordion", "harmonica", "sitar", "ukulele", "clarinet",
    "trombone", "tuba", "xylophone", "marimba", "vibraphone", "pipa", "koto",
    "bağlama", "kantele", "pan flute", "cowbell", "theremin", "vocoder",
}

VOCAL = {
    "male vocalist", "female vocalist", "vocal", "vocals", "backing vocals",
    "instrumental", "choir", "chant", "a cappella", "harmonies", "spoken",
    "rap vocals", "falsetto", "growl", "screaming", "whisper", "duet",
    "group vocals", "child vocals", "diction",
}

ERA = {
    "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
    "early 1950s", "early 1960s", "early 1970s", "early 1980s", "early 1990s",
    "early 2000s", "early 2010s", "late 1950s", "late 1960s", "late 1970s",
    "late 1980s", "late 1990s", "late 2000s", "late 2010s", "early 2020s",
    "late 2020s", "mid 2010s", "eighties", "nineties", "seventies", "sixties",
    "contemporary", "modern", "oldies", "retro", "vintage",
}

# The category vocabulary, in a stable order. Used for validation and for the
# UI's filter list, so an unknown category name is rejected rather than
# silently matching nothing.
CATEGORIES: tuple[str, ...] = ("mood", "genre", "instrument", "vocal", "era")

# --------------------------------------------------------------------------
# Query-side aliases: plain-English words people actually type -> real tags.
# This is what lets a user type "something sad and slow with piano" and get
# sane results with zero AI involvement.
# --------------------------------------------------------------------------

ALIASES: dict[str, list[str]] = {
    "chill": ["chill", "chillout", "mellow", "relaxation", "downtempo", "calm"],
    "relax": ["relaxation", "calm", "peaceful", "ambient", "soothing", "new age"],
    "study": ["ambient", "instrumental", "downtempo", "focus", "chillout"],
    "focus": ["ambient", "instrumental", "downtempo", "minimal"],
    "sleep": ["ambient", "binaural", "new age", "relaxation", "drone"],
    "workout": ["energetic", "hard rock", "edm", "hip hop", "trap", "metal", "dance"],
    "gym": ["energetic", "hard rock", "hip hop", "trap", "metal", "edm"],
    "party": ["party", "dance", "house", "edm", "disco", "techno", "hip hop"],
    "sad": ["sad", "melancholic", "slowcore", "sorrowful", "ballad", "emo"],
    "depressing": ["sad", "melancholic", "dsbm", "slowcore", "doom metal"],
    "happy": ["happy", "joyful", "upbeat", "cheerful", "pop"],
    "angry": ["angry", "aggressive", "hardcore", "punk", "metal", "rage"],
    "dark": ["dark", "dark ambient", "darkwave", "goth", "industrial", "ominous"],
    "epic": ["epic", "cinematic", "orchestral", "anthemic", "soundtrack"],
    "romantic": ["romantic", "love", "ballad", "sensual", "smooth jazz"],
    "sexy": ["sensual", "smooth", "r&b", "lounge", "slow jam"],
    "summer": ["summer", "tropical", "reggae", "surf", "bossa nova", "dance"],
    "winter": ["winter", "ambient", "christmas", "noël", "dark folk"],
    "christmas": ["christmas", "noël", "holiday", "carol"],
    "rain": ["rain", "ambient", "atmospheric", "downtempo"],
    "driving": ["driving", "rock", "synthpop", "highway", "classic rock"],
    "background": ["ambient", "lounge", "instrumental", "easy listening", "downtempo"],
    "dinner": ["lounge", "jazz", "bossa nova", "smooth jazz", "easy listening"],
    "coffee": ["jazz", "bossa nova", "lounge", "acoustic", "folk"],
    "retro": ["retro", "1980s", "1970s", "synthpop", "vintage", "oldies"],
    "80s": ["1980s", "synthpop", "new wave", "post-punk", "eighties"],
    "90s": ["1990s", "grunge", "hip hop", "nineties", "alternative rock"],
    "2000s": ["2000s", "pop punk", "nu metal", "emo"],
    "sing": ["vocal", "male vocalist", "female vocalist", "vocals"],
    "no vocals": ["instrumental"],
    "instrumental": ["instrumental"],
    "fast": ["energetic", "uptempo", "speed metal", "punk", "dance"],
    "slow": ["slowcore", "ballad", "downtempo", "ambient", "doom metal"],
    "loud": ["metal", "hard rock", "noise", "punk", "industrial"],
    "quiet": ["ambient", "acoustic", "soft", "instrumental", "folk"],
    "acoustic": ["acoustic", "unplugged", "folk", "singer-songwriter"],
    "electronic": ["electronic", "edm", "synth", "electro", "techno", "house"],
    "edm": ["edm", "house", "techno", "trance", "dubstep", "dance"],
    "lofi": ["lo-fi", "lofi", "chillhop", "barber beats", "downtempo"],
    "gaming": ["chiptune", "game arrange", "electronic", "synthwave", "epic"],
    "cinematic": ["cinematic", "soundtrack", "orchestral", "film", "score"],
    "guitar": ["guitar", "guitar solo", "acoustic guitar", "electric guitar"],
    "piano": ["piano", "piano solo", "classical", "baroque", "toccata"],
    "jazz": ["jazz", "contemporary jazz", "smooth jazz", "bebop", "swing"],
    "country": ["country", "americana", "bluegrass", "nashville sound/countrypolitan"],
    "hip hop": ["hip hop", "rap", "trap", "boom bap", "hip-hop/rap"],
    "metal": ["metal", "heavy metal", "death metal", "black metal", "thrash"],
    "punk": ["punk", "hardcore", "anarcho-punk", "pop punk", "post-punk"],
    "folk": ["folk", "folk rock", "americana", "singer-songwriter", "acoustic"],
    "classical": ["classical", "baroque", "romantic era", "opera", "orchestral"],
    "ambient": ["ambient", "drone", "atmospheric", "new age", "binaural"],
}

# Negative-intent words. "no drums", "without vocals", "not sad".
NEGATION_WORDS = {"no", "not", "without", "excluding", "except", "minus", "anti", "non"}

_STOPWORDS = {
    "i", "want", "wanna", "need", "some", "something", "like", "the", "a", "an",
    "and", "or", "with", "for", "that", "sounds", "sound", "music", "song",
    "songs", "track", "tracks", "give", "me", "please", "looking", "look",
    "find", "playlist", "vibe", "vibes", "feeling", "feel", "kind", "sort",
    "of", "to", "in", "on", "it", "is", "am", "be", "very", "really", "much",
    "make", "made", "would", "could", "should", "can", "get", "got", "my",
    "your", "you", "we", "they", "he", "she", "this", "these", "those",
    "more", "less", "best", "good", "great", "pretty", "quite", "just",
    "about", "into", "from", "up", "down", "out", "there", "here", "when",
    "what", "which", "who", "how", "why", "all", "any", "also", "but",
}


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse separators. 'Forró' -> 'forro'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    """
    Key used for identity/dedupe. Falls back to the raw text for tags that are
    entirely non-Latin (Cyrillic, Greek, Chinese, Devanagari...), which
    normalize() would otherwise flatten to an empty string and lose.
    """
    n = normalize(text)
    if n:
        return n
    return unicodedata.normalize("NFC", text).lower().strip()


def tokenize(text: str) -> list[str]:
    """
    Tokens for the inverted index. Latin text splits on spaces; scripts with no
    spaces (Chinese, Japanese) also get per-character tokens so they stay
    searchable.
    """
    n = normalize(text)
    if n:
        return n.split()
    base = unicodedata.normalize("NFC", text).lower()
    chars = [c for c in base if not c.isspace() and not unicodedata.combining(c)]
    return chars


@dataclass
class Tag:
    name: str
    norm: str
    categories: tuple[str, ...] = ()
    popularity: int = 0  # 1 = most popular (file is popularity-ordered)
    tokens: tuple[str, ...] = ()

    @property
    def score_weight(self) -> float:
        """Earlier in the file = more common = slight boost, but bounded."""
        return 1.0 + (0.35 * (1.0 - min(self.popularity, 4000) / 4000.0))


class TagIndex:
    """
    Loads tags once and answers search queries fast.

    Beyond the raw vocabulary this builds two derived structures, both learned
    from the tag list itself with no external data:

      - a BM25 corpus (document frequencies, lengths) for principled ranking
      - a co-occurrence graph plus a nearest-centroid classifier, used to
        interpret words that are not tags without any network call

    The derived build is optional (build_derived=False) so tests and the CLI
    can load the vocabulary alone when they only need lookups.
    """

    def __init__(self, path: str | Path, build_derived: bool = True):
        self.path = Path(path)
        self.tags: list[Tag] = []
        self._by_norm: dict[str, Tag] = {}
        self._tokens: dict[str, set[int]] = {}
        self._sorted_tokens: list[str] = []
        self._alias_norm: dict[str, list[str]] = {}
        self._sorted_aliases: list[str] = []
        self.corpus = None          # ranking.Corpus
        self.graph = None           # intent.TagGraph
        self.classifier = None      # intent.CentroidClassifier
        self.intent_parser = None   # intent.IntentParser
        self._load()
        if build_derived:
            self.build_derived()

    # ---------------- loading ----------------

    def _load(self) -> None:
        raw = self.path.read_text(encoding="utf-8", errors="replace")
        seen: set[str] = set()
        for i, line in enumerate(raw.splitlines()):
            name = line.strip()
            if not name:
                continue
            key = norm_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            norm = normalize(name)
            toks = tuple(tokenize(name))
            tag = Tag(
                name=name,
                norm=norm or key,
                categories=self._categorize(norm or key),
                popularity=i + 1,
                tokens=toks,
            )
            self.tags.append(tag)
            self._by_norm[tag.norm] = tag
            for tok in toks:
                self._tokens.setdefault(tok, set()).add(len(self.tags) - 1)

        # Sorted token list for fast prefix expansion (bisect instead of a
        # full scan on every query word).
        self._sorted_tokens = sorted(self._tokens)

        for key, vals in ALIASES.items():
            self._alias_norm[normalize(key)] = [normalize(v) for v in vals]
        # Sorted alias keys for fast prefix lookup.
        self._sorted_aliases = sorted(self._alias_norm)

        import ranking
        self.corpus = ranking.build_corpus([list(t.tokens) for t in self.tags])

    def build_derived(self) -> None:
        """
        Build the graph, classifier and intent parser. Imported lazily so
        intent.py can import from this module without a circular import.
        """
        if self.graph is not None:
            return
        import intent as intent_mod

        self.graph = intent_mod.TagGraph(index=self)
        self.graph.build()
        self.classifier = intent_mod.CentroidClassifier(index=self)
        self.classifier.train()
        self.intent_parser = intent_mod.IntentParser(self, self.graph, self.classifier)

    @property
    def has_derived(self) -> bool:
        return self.graph is not None

    @staticmethod
    def _categorize(norm: str) -> tuple[str, ...]:
        cats = []
        if norm in MOOD:
            cats.append("mood")
        if norm in GENRE:
            cats.append("genre")
        if norm in INSTRUMENT:
            cats.append("instrument")
        if norm in VOCAL:
            cats.append("vocal")
        if norm in ERA:
            cats.append("era")
        return tuple(cats)

    # ---------------- introspection ----------------

    def __len__(self) -> int:
        return len(self.tags)

    def exists(self, name: str) -> bool:
        return norm_key(name) in self._by_norm

    def get(self, name: str) -> Tag | None:
        return self._by_norm.get(norm_key(name))

    def by_category(self, cat: str, limit: int = 200) -> list[Tag]:
        return [t for t in self.tags if cat in t.categories][:limit]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.tags:
            for c in t.categories:
                counts[c] = counts.get(c, 0) + 1
        out = {
            "total": len(self.tags),
            "categories": counts,
            "avg_tokens_per_tag": round(self.corpus.avgdl, 2) if self.corpus else 0,
            "vocabulary_terms": len(self._tokens),
            "derived_built": self.has_derived,
        }
        if self.graph is not None:
            out["graph"] = self.graph.token_stats()
        return out

    def related(self, tag: str, limit: int = 10) -> list[dict]:
        """
        Graph neighbours of a tag, with the shared tokens that explain the
        relationship. Learned, not hand-written.
        """
        if self.graph is None:
            return []
        base = self.get(tag)
        base_tokens = set(base.tokens) if base else set()
        out = []
        for other, weight in self.graph.related(tag, limit=limit):
            t = self._by_norm.get(other)
            if not t:
                continue
            out.append({
                "tag": t.name,
                "weight": round(weight, 3),
                "shared": sorted(base_tokens & set(t.tokens)),
                "categories": list(t.categories),
            })
        return out

    def similar(self, tag: str, limit: int = 5) -> list[str]:
        """Near-synonym tags (heavy token overlap)."""
        if self.graph is None:
            return []
        return self.graph.similar(tag, limit=limit)

    def interpret(self, query: str):
        """Structured reading of a query. None if derived structures are off."""
        if self.intent_parser is None:
            return None
        return self.intent_parser.parse(query)

    # ---------------- search ----------------

    def _prefix_tokens(self, prefix: str, cap: int = 300) -> list[str]:
        """All indexed tokens starting with `prefix`, via bisect. Bounded."""
        if len(prefix) < 3:
            return []
        import bisect

        i = bisect.bisect_left(self._sorted_tokens, prefix)
        out: list[str] = []
        while i < len(self._sorted_tokens) and self._sorted_tokens[i].startswith(prefix):
            out.append(self._sorted_tokens[i])
            i += 1
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _stems(token: str) -> list[str]:
        """Cheap English stems so 'studying' finds 'study' and 'dancing' finds 'dance'."""
        out: list[str] = []
        if len(token) > 4:
            if token.endswith("ing"):
                base = token[:-3]
                out.append(base)
                out.append(base + "e")
                if len(base) > 2 and base[-1] == base[-2]:
                    out.append(base[:-1])  # running -> run
            elif token.endswith("ies"):
                out.append(token[:-3] + "y")
            elif token.endswith("ed"):
                out.append(token[:-2])
                out.append(token[:-1])
            elif token.endswith("s") and not token.endswith("ss"):
                out.append(token[:-1])
        return [s for s in out if len(s) >= 3]

    def _alias_expand(self, token: str) -> list[str]:
        """Expand one query token through the alias table (prefix-tolerant)."""
        out: list[str] = []
        if token in self._alias_norm:
            out.extend(self._alias_norm[token])
        else:
            # bounded prefix scan over alias keys only (small dict)
            import bisect

            i = bisect.bisect_left(self._sorted_aliases, token)
            while i < len(self._sorted_aliases) and self._sorted_aliases[i].startswith(token):
                out.extend(self._alias_norm[self._sorted_aliases[i]])
                i += 1
            # stems: studying -> study, dancing -> dance, parties -> party
            for stem in self._stems(token):
                if stem in self._alias_norm:
                    out.extend(self._alias_norm[stem])
        return out

    def search(
        self,
        query: str,
        limit: int = 40,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        categories: Iterable[str] = (),
        fuzzy: bool = True,
        use_graph: bool = True,
        explain: bool = False,
    ) -> list[dict]:
        """
        BM25 search over the vocabulary with learned expansion.

        Ranking is principled rather than hand-tuned: a term appearing in 2
        tags carries far more weight than one appearing in 800, and BM25's
        inverse document frequency computes that automatically.

        Returns [{tag, score, confidence, matched_on, categories, parts?}]
        ranked best first, with `confidence` rescaled to 0..1 for display.
        """
        import ranking

        q_norm = normalize(query)
        want_cats = {c for c in categories if c in CATEGORIES}
        inc = [normalize(x) for x in include if normalize(x)]
        exc = [normalize(x) for x in exclude if normalize(x)]

        # --- parse negation out of the free-text query: "no drums", "without vocals"
        tokens = q_norm.split()
        pos_tokens: list[str] = []
        neg_tokens: list[str] = []
        skip_next = False
        for tok in tokens:
            if skip_next:
                neg_tokens.append(tok)
                skip_next = False
                continue
            if tok in NEGATION_WORDS:
                skip_next = True
                continue
            pos_tokens.append(tok)
        exc = list(dict.fromkeys(exc + neg_tokens))

        # --- expand positives through aliases, then the graph for unknown words
        expanded: dict[str, float] = {}  # normalized phrase -> weight
        for tok in pos_tokens:
            if tok in _STOPWORDS or len(tok) < 2:
                continue
            expanded[tok] = max(expanded.get(tok, 0.0), 1.0)
            for alias in self._alias_expand(tok):
                expanded[alias] = max(expanded.get(alias, 0.0), 0.85)
            # A word that is not a tag at all still has meaning: reach it
            # through morphology and co-occurrence instead of giving up.
            if use_graph and self.graph is not None and not self.exists(tok):
                for other in self.graph.expand_token(tok, limit=8):
                    expanded[other] = max(expanded.get(other, 0.0), 0.6)
        for phrase in inc:
            expanded[phrase] = max(expanded.get(phrase, 0.0), 1.0)
            for alias in self._alias_expand(phrase):
                expanded[alias] = max(expanded.get(alias, 0.0), 0.85)

        # whole-query phrase bonus
        phrase_key = q_norm if len(tokens) > 1 else ""

        if not expanded and not want_cats:
            return []

        # --- candidate gathering
        candidates: set[int] = set()
        for key in expanded:
            for tok in key.split():
                candidates |= self._tokens.get(tok, set())
                # prefix matching on tokens so "psyche" finds psychedelic
                for pt in self._prefix_tokens(tok):
                    candidates |= self._tokens.get(pt, set())
        if want_cats:
            candidates |= {i for i, t in enumerate(self.tags) if want_cats & set(t.categories)}
        if not candidates:
            candidates = set(range(len(self.tags)))

        corpus = self.corpus

        # --- score
        results: list[dict] = []
        for i in candidates:
            tag = self.tags[i]
            if want_cats and not (want_cats & set(tag.categories)):
                continue
            if any(self._is_excluded(tag.norm, e) for e in exc):
                continue

            parts = ranking.ScoreParts()
            matched: list[str] = []
            doc_len = len(tag.tokens) or 1
            tag_norm = tag.norm
            tag_tokens = tag.tokens

            for key, weight in expanded.items():
                key_tokens = key.split()

                if tag_norm == key:
                    parts.exact += ranking.W_EXACT * weight
                    matched.append(key)
                    continue

                if len(key_tokens) > 1 and key in tag_norm:
                    parts.phrase += ranking.W_PHRASE * weight
                    matched.append(key)
                    continue

                hit = [t for t in key_tokens if t in tag_tokens]
                if hit:
                    parts.bm25 += ranking.bm25_term_score(
                        len(hit), doc_len, corpus, key_tokens[0], ranking.W_TOKEN
                    ) * weight
                    matched.append(key)
                    continue

                if any(len(t) >= 3 and any(tok.startswith(t) for tok in tag_tokens)
                       for t in key_tokens):
                    parts.prefix += ranking.W_PREFIX * weight
                    matched.append(f"{key}*")
                    continue

                if fuzzy and len(key) >= 4 and self._fuzzy_contains(tag_norm, key):
                    parts.fuzzy += ranking.W_FUZZY * weight
                    matched.append(f"~{key}")

            if not matched:
                continue

            if phrase_key and phrase_key in tag_norm:
                parts.phrase += 2.0
            if tag.categories:
                parts.category += 0.35 * len(tag.categories)
            parts.popularity += ranking.popularity_boost(tag.popularity, len(self.tags))

            score = parts.total
            if score <= 0:
                continue

            row = {
                "tag": tag.name,
                "score": round(score, 3),
                "matched_on": sorted(set(matched)),
                "categories": list(tag.categories),
            }
            if explain:
                row["parts"] = parts.breakdown()
            results.append(row)

        results.sort(key=lambda r: (-r["score"], r["tag"]))
        results = results[:limit]
        return ranking.normalize_scores(results)

    def _is_excluded(self, tag_norm: str, excl: str) -> bool:
        if not excl:
            return False
        if tag_norm == excl:
            return True
        if excl in tag_norm.split():
            return True
        # 'no drums' should kill 'drums (drum set)' but not 'drum and bass'
        if excl.endswith("s") and excl[:-1] in tag_norm.split():
            return True
        return False

    @staticmethod
    def _fuzzy_contains(haystack: str, needle: str) -> bool:
        if needle in haystack:
            return True
        for word in haystack.split():
            if len(word) >= 4 and SequenceMatcher(None, word, needle).ratio() >= 0.82:
                return True
        return False

    def suggest(self, prefix: str, limit: int = 12) -> list[str]:
        p = normalize(prefix)
        if not p:
            return [t.name for t in self.tags[:limit]]
        starts = [t.name for t in self.tags if t.norm.startswith(p)]
        contains = [t.name for t in self.tags if p in t.norm and not t.norm.startswith(p)]
        return (starts + contains)[:limit]


# --------------------------------------------------------------------------
# Related-tag expansion: used to turn a shortlist into a usable palette.
# --------------------------------------------------------------------------

# Mood tags that only make sense for certain energies. Without this the palette
# cheerfully offers "melancholic" to someone asking for party music.
BRIGHT_MOODS = {"happy", "joyful", "uplifting", "euphoric", "party", "energetic",
                "cheerful", "playful", "triumphant", "groovy", "dance", "funky"}
DARK_MOODS = {"sad", "melancholic", "dark", "sorrowful", "somber", "brooding",
              "lonely", "bittersweet", "haunting", "menacing", "tense", "anxious"}
CALM_MOODS = {"calm", "peaceful", "relaxation", "ambient", "soothing", "serene",
              "tranquil", "meditative", "dreamy", "mellow", "chill", "soft"}


def mood_family(tags: list[str]) -> str:
    """Classify the tone of a set of seed tags: bright, dark, calm, or mixed."""
    n = {normalize(t) for t in tags}
    hits = {
        "bright": len(n & BRIGHT_MOODS),
        "dark": len(n & DARK_MOODS),
        "calm": len(n & CALM_MOODS),
    }
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] > 0 else "mixed"


# Mood suggestions per family, so the palette stays on-tone.
MOOD_BY_FAMILY: dict[str, list[str]] = {
    "bright": ["uplifting", "energetic", "playful", "euphoric", "cheerful", "groovy"],
    "dark": ["melancholic", "brooding", "haunting", "somber", "bittersweet", "tense"],
    "calm": ["dreamy", "peaceful", "soothing", "mellow", "serene", "meditative"],
    "mixed": ["hopeful", "dreamy", "nostalgic", "atmospheric", "introspective", "moody"],
}


def build_palette(index: TagIndex, seed_tags: list[str], per_bucket: int = 3) -> dict[str, list[str]]:
    """
    Related suggestions, grouped by category.

    Mood picks come from the tone family so a party query never gets offered
    "melancholic". Other categories come from the learned graph when it is
    available, which means the suggestions reflect the actual vocabulary
    instead of a list I typed out.
    """
    out: dict[str, list[str]] = {}
    seed_norms = {normalize(s) for s in seed_tags}
    seed_cats: set[str] = set()
    for s in seed_tags:
        t = index.get(s)
        if t:
            seed_cats |= set(t.categories)

    family = mood_family(seed_tags)
    for cat in CATEGORIES:
        if seed_cats and cat not in seed_cats:
            continue

        picks: list[str] = []
        if cat == "mood":
            picks = [m for m in MOOD_BY_FAMILY[family] if index.exists(m)]
        elif index.graph is not None:
            # Graph neighbours that live in this category.
            for s in seed_tags:
                for other in index.related(s, limit=12):
                    if other["categories"] and cat in other["categories"]:
                        picks.append(other["tag"])

        if not picks:
            continue

        deduped: list[str] = []
        for p in picks:
            k = normalize(p) or p
            if k not in seed_norms and p not in deduped:
                deduped.append(p)
        if deduped:
            out[cat] = deduped[:per_bucket]
    return out
