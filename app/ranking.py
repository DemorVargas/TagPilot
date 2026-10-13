"""
Ranking: BM25 over the tag vocabulary, with field weighting.

The previous version used hand-tuned magic numbers (exact match = 6.0, word
match = 3.2, ...). That works but it is not principled and it does not account
for how rare a term is. BM25 fixes both: a term that appears in 2 tags is far
more informative than one that appears in 800, and BM25 weights it that way
automatically via inverse document frequency.

On a short-document corpus like tags, BM25's length normalisation needs a low
`b`, because tags are 1-4 tokens and length differences are mostly meaningful
rather than noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# BM25 parameters, tuned for very short documents.
K1 = 1.2      # term frequency saturation
B = 0.35      # length normalisation (low: tags are short and length matters)

# Raw BM25 values are unbounded (a rare single term can reach ~13 on this
# corpus) which would drown out the structural signals below. Scaling brings
# them onto the same footing: BM25 measures "how well do the words match",
# the weights below measure "how strong is this kind of match".
BM25_SCALE = 0.30

# Field weights. A query term matching the whole tag is a much stronger signal
# than the same term matching one word inside a long tag, so these are set
# high enough to outrank a plain token match.
W_EXACT = 6.0     # whole tag equals the query term
W_PHRASE = 3.5    # query term is a contiguous phrase in the tag
W_TOKEN = 1.0     # query term matches one token (multiplied into BM25)
W_PREFIX = 0.60   # query term is a prefix of a token
W_FUZZY = 0.30    # near-miss spelling


@dataclass
class Corpus:
    """Document statistics BM25 needs, computed once at load."""

    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 1.0
    df: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0

    def idf(self, term: str) -> float:
        """
        Robertson-Sparck-Jones IDF with the +0.5 smoothing that keeps it
        positive for terms appearing in most documents.
        """
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))


def build_corpus(tokenized_docs: list[list[str]]) -> Corpus:
    c = Corpus()
    c.n_docs = len(tokenized_docs)
    c.doc_len = [len(d) for d in tokenized_docs]
    total = sum(c.doc_len)
    c.avgdl = (total / c.n_docs) if c.n_docs else 1.0
    df: dict[str, int] = {}
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    c.df = df
    return c


def bm25_term_score(
    tf: int,
    doc_len: int,
    corpus: Corpus,
    term: str,
    field_weight: float = W_TOKEN,
) -> float:
    """
    BM25 contribution of one query term to one document, scaled onto the same
    footing as the structural weights in ScoreParts.
    """
    if tf <= 0 or corpus.n_docs == 0:
        return 0.0
    idf = corpus.idf(term)
    denom = tf + K1 * (1.0 - B + B * (doc_len / (corpus.avgdl or 1.0)))
    if denom <= 0:
        return 0.0
    return idf * ((tf * (K1 + 1.0)) / denom) * field_weight * BM25_SCALE


# --------------------------------------------------------------------------
# Score combination
# --------------------------------------------------------------------------

@dataclass
class ScoreParts:
    """Why a tag scored what it did. Kept for transparency in the UI."""

    bm25: float = 0.0
    exact: float = 0.0
    phrase: float = 0.0
    prefix: float = 0.0
    fuzzy: float = 0.0
    category: float = 0.0
    popularity: float = 0.0

    @property
    def total(self) -> float:
        return (self.bm25 + self.exact + self.phrase + self.prefix
                + self.fuzzy + self.category + self.popularity)

    def breakdown(self) -> dict:
        return {
            k: round(v, 3)
            for k, v in (
                ("bm25", self.bm25),
                ("exact", self.exact),
                ("phrase", self.phrase),
                ("prefix", self.prefix),
                ("fuzzy", self.fuzzy),
                ("category", self.category),
                ("popularity", self.popularity),
            )
            if v
        }


def popularity_boost(rank: int, corpus_size: int, max_boost: float = 0.8) -> float:
    """
    Nudge common tags up. Bounded and small: it breaks ties between equally
    relevant tags, it does not decide relevance.
    """
    if corpus_size <= 0:
        return 0.0
    frac = 1.0 - min(rank, corpus_size) / corpus_size
    return max_boost * frac


def normalize_scores(results: list[dict], floor: float = 0.0) -> list[dict]:
    """
    Rescale scores to 0..1 for display, preserving order. Raw BM25 values are
    unbounded and meaningless to a human; a 0-1 confidence is not.
    """
    if not results:
        return results
    top = max(r["score"] for r in results)
    if top <= 0:
        return results
    for r in results:
        r["confidence"] = round(max(floor, r["score"] / top), 4)
    return results
