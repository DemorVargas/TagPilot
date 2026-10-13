"""
Offline understanding: the vocabulary becomes its own knowledge base.

The problem with the v1 engine is that a query like "music for a rainy
sunday morning" only works because I hand-wrote an alias for it. Hand-written
aliases do not scale and they only cover what I thought of.

This module derives relationships from the data instead:

  1. **Morphology**  - "psychedelic", "psychedelia", "psytrance" share a stem.
                       Learned from the vocabulary, no dictionary.
  2. **Co-occurrence** - tags that share tokens are neighbours. "dark ambient"
                       and "dark jazz" are related because they share "dark",
                       and that relationship is computed, not declared.
  3. **Learned synonyms** - two tags are near-synonyms when their token sets
                       heavily overlap ("hip hop" / "hip-hop/rap").
  4. **Centroid classification** - an unknown word is placed in mood / genre /
                       instrument space by comparing it to the centroid of the
                       known members of each class.

Everything here runs on CPU in milliseconds with no network and no model
weights. This is the "offline AI" in the honest sense: real inference over a
learned structure, not a stub.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tags import normalize, tokenize

# --------------------------------------------------------------------------
# Morphology: learn stems from the vocabulary itself
# --------------------------------------------------------------------------

# Suffixes that mark a derivational relationship between tags.
_DERIV_SUFFIXES = (
    "ical", "ically", "ness", "ment", "tion", "sion", "ing", "ish", "ive",
    "ous", "ial", "ance", "ence", "ity", "ism", "ist", "ed", "er", "ly", "s",
)


def stem_word(word: str) -> str:
    """
    Crude but effective suffix stripper.

    Deliberately not a real stemmer: over-stemming is worse than under-stemming
    here because we only use this to link tags, and a false link between
    "rock" and "rocket" is more damaging than a missed link.

    Order matters. Plural reduction runs first so "psychedelics" becomes
    "psychedelic" and then "psychedel", rather than stopping at "psychedelic"
    and failing to meet "psychedelia". Derivational suffixes are then checked
    longest-first, because "metallic" must reduce to "metal" via "-lic", not to
    "metall" via "-ic".
    """
    w = word
    if len(w) <= 4:
        return w

    # --- 1. plural reduction
    if w.endswith("ies") and len(w) - 3 >= 3:
        w = w[:-3] + "y"
    elif w.endswith("es") and len(w) - 2 >= 4 and not w.endswith("ses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) - 1 >= 3:
        w = w[:-1]

    # --- 2. derivational suffixes, longest first.
    # Note "lic" and "tic" are deliberately absent: they turn "psychedelic"
    # into "psychede". The doubled-consonant collapse below handles the case
    # they were there for ("metallic" -> "metall" -> "metal").
    for suf in ("ically", "ical", "ica", "ic", "ial", "ia",
                "ness", "ment", "tion", "sion", "ing", "ish", "ive",
                "ous", "ance", "ence", "ity", "ism", "ist", "ed", "er", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    else:
        return w

    # --- 3. collapse a doubled trailing consonant: metall -> metal
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]

    return w


# --------------------------------------------------------------------------
# Co-occurrence graph
# --------------------------------------------------------------------------

@dataclass
class TagGraph:
    """
    An undirected weighted graph over tags, built purely from shared tokens.

    Edge weight is a cosine-style overlap, which is cheap and needs no
    training. It answers "what is near this tag?" and "what does this word
    probably mean?" without any external knowledge.
    """

    index: object  # TagIndex
    neighbours: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    token_df: Counter = field(default_factory=Counter)
    stem_to_tags: dict[str, set[str]] = field(default_factory=dict)
    token_to_tags: dict[str, set[str]] = field(default_factory=dict)
    _built: bool = False

    def build(self) -> None:
        if self._built:
            return
        tag_tokens: dict[str, set[str]] = {}

        for t in self.index.tags:
            toks = set(tokenize(t.name))
            toks |= {stem_word(x) for x in toks}
            tag_tokens[t.norm] = toks
            for tok in toks:
                self.token_df[tok] += 1
                self.token_to_tags.setdefault(tok, set()).add(t.norm)
                self.stem_to_tags.setdefault(stem_word(tok), set()).add(t.norm)

        # Invert: token -> tags, then score tag pairs by shared-token overlap.
        # Only tokens that are not ubiquitous produce useful edges, so we skip
        # tokens present in a large fraction of the vocabulary.
        n = len(self.index.tags) or 1
        max_df = max(50, int(n * 0.02))
        informative = {tok for tok, df in self.token_df.items() if 1 < df <= max_df}

        pair_scores: dict[str, Counter] = defaultdict(Counter)
        for tok in informative:
            owners = self.token_to_tags.get(tok, set())
            if len(owners) > 400:  # keep it bounded
                continue
            owners = list(owners)
            for i, a in enumerate(owners):
                for b in owners[i + 1:]:
                    pair_scores[a][b] += 1.0 / math.log(2 + self.token_df[tok])
                    pair_scores[b][a] += 1.0 / math.log(2 + self.token_df[tok])

        for a, counter in pair_scores.items():
            self.neighbours[a] = counter.most_common(24)

        self._tag_tokens = tag_tokens
        self._built = True

    # ----------------------------------------------------------------------

    def related(self, tag: str, limit: int = 8) -> list[tuple[str, float]]:
        self.build()
        key = normalize(tag) or tag
        out = self.neighbours.get(key, [])
        return out[:limit]

    def similar(self, tag: str, limit: int = 5) -> list[str]:
        """Near-synonyms: heaviest overlap, usually the same tag spelled differently."""
        self.build()
        key = normalize(tag) or tag
        mine = self._tag_tokens.get(key, set())
        if not mine:
            return []
        scored = []
        for other, _ in self.neighbours.get(key, []):
            theirs = self._tag_tokens.get(other, set())
            if not theirs:
                continue
            jac = len(mine & theirs) / max(1, len(mine | theirs))
            if jac >= 0.5:
                scored.append((jac, other))
        scored.sort(reverse=True)
        names = []
        for _, other in scored[:limit]:
            t = self.index.get(other)
            if t:
                names.append(t.name)
        return names

    def expand_token(self, token: str, limit: int = 12) -> list[str]:
        """
        Given a bare word, find tags that use it, its stem, or a stem sibling.

        This is how "psychedelia" reaches "psychedelic rock" with no alias
        table entry: both reduce to the stem "psychedel".
        """
        self.build()
        hits: Counter = Counter()
        if token in self.token_to_tags:
            for t in self.token_to_tags[token]:
                hits[t] += 2.0
        st = stem_word(token)
        for t in self.stem_to_tags.get(st, set()):
            hits[t] += 1.0
        # stem siblings of the token's own tokens
        for tok in self.token_to_tags.get(token, set()):
            for t in self.stem_to_tags.get(stem_word(tok), set()):
                hits[t] += 0.25
        return [h for h, _ in hits.most_common(limit)]

    def token_stats(self) -> dict:
        self.build()
        return {
            "tokens": len(self.token_df),
            "stems": len(self.stem_to_tags),
            "tags_with_neighbours": len(self.neighbours),
        }


# --------------------------------------------------------------------------
# Centroid classifier: place an unknown word into a category
# --------------------------------------------------------------------------

@dataclass
class CentroidClassifier:
    """
    Nearest-centroid classifier over tag names, trained on the categories the
    index already knows (mood / genre / instrument / vocal / era).

    Used to answer "the user typed a word that is not a tag, which kind of tag
    were they probably after?" and to bias retrieval toward that class. Pure
    arithmetic, no model file, no network.
    """

    index: object
    centroids: dict[str, Counter] = field(default_factory=dict)
    _trained: bool = False

    def train(self) -> None:
        if self._trained:
            return
        for cat in ("mood", "genre", "instrument", "vocal", "era"):
            counts: Counter = Counter()
            members = self.index.by_category(cat, limit=5000)
            for t in members:
                for tok in tokenize(t.name):
                    counts[tok] += 1
                    counts[stem_word(tok)] += 1
            # L2-normalise so big categories do not dominate by mass alone.
            norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
            self.centroids[cat] = Counter({k: v / norm for k, v in counts.items()})
        self._trained = True

    def classify(self, word: str, top: int = 3) -> list[tuple[str, float]]:
        """
        Return [(category, similarity)] best first, cosine similarity.

        The word is represented by its tokens, their stems, and its prefix
        (so a word the vocabulary has never seen can still be placed by how it
        is built).
        """
        self.train()
        toks = [w for w in tokenize(word)] or [word]
        vec: Counter = Counter()
        for tok in toks:
            vec[tok] += 1.0
            vec[stem_word(tok)] += 1.0
            # Prefix evidence: "psychedelia" -> "psyched" appears in
            # "psychedelic rock", so the category bleeds through.
            if len(tok) >= 6:
                vec[tok[:6]] += 0.5
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        scores = []
        for cat, cent in self.centroids.items():
            dot = sum(v * cent.get(k, 0.0) for k, v in vec.items())
            scores.append((cat, dot / norm))
        scores.sort(key=lambda x: -x[1])
        return scores[:top]

    def confidence(self, word: str) -> float:
        """How sure are we that this word belongs to any known class?"""
        got = self.classify(word, top=1)
        return got[0][1] if got else 0.0


# --------------------------------------------------------------------------
# Query understanding without a model
# --------------------------------------------------------------------------

# Words that signal the user is describing a feeling rather than a genre.
_FEELING_MARKERS = {
    "feel", "feels", "feeling", "vibe", "vibes", "mood", "moody", "emotional",
    "emotion", "atmosphere", "atmospheric", "energy", "energetic", "tone",
    "like", "similar", "reminds", "remind", "nostalgic", "bittersweet",
}

# Words that signal a use case, which maps onto functional tags.
_USE_CASE_MARKERS = {
    "study", "studying", "work", "working", "focus", "sleep", "sleeping",
    "gym", "workout", "running", "driving", "dinner", "party", "cleaning",
    "cooking", "reading", "relax", "relaxing", "meditation", "yoga",
}

_ENERGY_HIGH = {"fast", "loud", "energetic", "upbeat", "aggressive", "intense",
                "hard", "heavy", "driving", "pumping", "hype", "banging"}
_ENERGY_LOW = {"slow", "quiet", "soft", "gentle", "calm", "chill", "mellow",
               "relaxing", "sleepy", "ambient", "peaceful", "tranquil"}


@dataclass
class QueryIntent:
    """Structured reading of a free-text query. All derived, all explainable."""

    raw: str = ""
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    energy: str = "neutral"          # high | low | neutral
    wants_vocals: str = "any"        # yes | no | any
    use_case: str = ""
    feeling: bool = False
    categories: list[str] = field(default_factory=list)
    unknown_words: list[str] = field(default_factory=list)
    guessed_categories: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        bits = []
        if self.energy != "neutral":
            bits.append(f"{self.energy} energy")
        if self.wants_vocals == "no":
            bits.append("instrumental")
        elif self.wants_vocals == "yes":
            bits.append("with vocals")
        if self.use_case:
            bits.append(f"for {self.use_case}")
        if self.feeling:
            bits.append("mood-led")
        if self.categories:
            bits.append("focus on " + "/".join(self.categories))
        return ", ".join(bits) if bits else "general"


class IntentParser:
    """
    Read a sentence into a structured intent using the vocabulary, the graph,
    and the classifier. No network, no model, no alias table required.
    """

    def __init__(self, index, graph: TagGraph, classifier: CentroidClassifier):
        self.index = index
        self.graph = graph
        self.classifier = classifier
        self.graph.build()
        self.classifier.train()

    def parse(self, query: str) -> QueryIntent:
        intent = QueryIntent(raw=query)
        q = normalize(query)
        if not q:
            return intent

        toks = q.split()
        negated: set[int] = set()
        for i, tok in enumerate(toks):
            if tok in ("no", "not", "without", "except", "excluding", "minus", "anti", "non"):
                if i + 1 < len(toks):
                    negated.add(i + 1)
                if i + 2 < len(toks):
                    negated.add(i + 2)

        # Words that carry no retrieval signal. Imported from tags so the two
        # modules agree on what a stopword is.
        from tags import _STOPWORDS

        for i, tok in enumerate(toks):
            if len(tok) < 2:
                continue
            if i in negated:
                intent.negative.append(tok)
                continue

            if tok in _STOPWORDS:
                # Still read the intent signal, just do not treat it as a term.
                if tok in _ENERGY_HIGH:
                    intent.energy = "high"
                elif tok in _ENERGY_LOW:
                    intent.energy = "low"
                if tok in _USE_CASE_MARKERS:
                    intent.use_case = tok.rstrip("ing") if tok.endswith("ing") else tok
                if tok in _FEELING_MARKERS:
                    intent.feeling = True
                continue

            intent.positive.append(tok)

            if tok in _ENERGY_HIGH:
                intent.energy = "high"
            elif tok in _ENERGY_LOW:
                intent.energy = "low"
            if tok in _USE_CASE_MARKERS:
                intent.use_case = tok.rstrip("ing") if tok.endswith("ing") else tok
            if tok in _FEELING_MARKERS:
                intent.feeling = True

        # Vocals intent.
        if any(t in ("instrumental", "instrumentals", "karaoke", "backing") for t in intent.positive):
            intent.wants_vocals = "no"
        if any(t in ("sing", "singing", "vocals", "vocal", "singer", "lyrics", "words")
               for t in intent.positive):
            intent.wants_vocals = "yes"
        if "instrumental" in intent.negative:
            intent.wants_vocals = "yes"
        if any(t in ("vocals", "vocal", "singing", "singer") for t in intent.negative):
            intent.wants_vocals = "no"

        # Which words are real tags, which the graph can reach, and which are
        # genuinely unrecognised?
        for tok in intent.positive:
            if self.index.exists(tok):
                t = self.index.get(tok)
                for c in t.categories:
                    if c not in intent.categories:
                        intent.categories.append(c)
                continue

            if len(tok) < 4:
                continue

            # Can morphology or co-occurrence reach it? If so it is understood,
            # even though it is not a tag itself.
            reachable = self.graph.expand_token(tok, limit=1) if self.graph else []

            cats = self.classifier.classify(tok, top=1)
            if cats and cats[0][1] > 0.12:
                intent.guessed_categories[tok] = cats[0][0]
            elif not reachable:
                intent.unknown_words.append(tok)

        return intent


# --------------------------------------------------------------------------
# Functional tags: what to add for a use case, derived not hardcoded where
# possible. These are seeds; the graph supplies the rest.
# --------------------------------------------------------------------------

USE_CASE_SEEDS: dict[str, list[str]] = {
    "study": ["ambient", "instrumental", "downtempo"],
    "work": ["instrumental", "ambient", "lofi"],
    "focus": ["ambient", "minimal", "drone"],
    "sleep": ["ambient", "binaural", "new age"],
    "gym": ["energetic", "hard rock", "trap"],
    "workout": ["energetic", "edm", "hip hop"],
    "running": ["energetic", "dance", "electronic"],
    "driving": ["rock", "synthpop", "classic rock"],
    "dinner": ["jazz", "bossa nova", "lounge"],
    "party": ["dance", "house", "disco"],
    "cleaning": ["pop", "dance", "funk"],
    "cooking": ["jazz", "soul", "bossa nova"],
    "reading": ["ambient", "classical", "piano"],
    "relax": ["ambient", "chillout", "downtempo"],
    "meditation": ["ambient", "new age", "drone"],
    "yoga": ["ambient", "new age", "world"],
}


def use_case_tags(intent: QueryIntent, index, graph: TagGraph) -> list[str]:
    """
    Expand a use case into real tags, using the graph to reach further than the
    seed list. Only returns tags that actually exist.
    """
    out: list[str] = []
    seeds = USE_CASE_SEEDS.get(intent.use_case, [])
    for s in seeds:
        if index.exists(s):
            out.append(s)
            for other, weight in graph.related(s, limit=3):
                if weight > 0.4:
                    t = index.get(other)
                    if t:
                        out.append(t.name)
    # Dedupe, keep order, bound.
    seen: set[str] = set()
    final: list[str] = []
    for t in out:
        k = normalize(t) or t
        if k not in seen:
            seen.add(k)
            final.append(t)
    return final[:10]
