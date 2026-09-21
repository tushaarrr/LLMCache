"""embed -> search -> threshold -> answer.

semcache is not an LLM SDK wrapper. You call your own model; this only decides
whether you need to.
"""

import re

import numpy as np

from .store import Store

# GPTCache's default translated. It searches with faiss L2 distance over
# normalized vectors and accepts distance <= 0.8; for normalized vectors
# L2^2 = 2 - 2*cos, so 0.8^2 = 2 - 2*cos gives cos = 0.68.
DEFAULT_THRESHOLD = 0.68
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Prompts carrying an opaque identifier are not consulted against the cache at
# all. This is not tuning -- it is the only thing that works. A general embedder
# scores "ticket ENG-4471" against "ticket ENG-4472" at 0.99: the digits are
# near-noise to it, so NO threshold separates them. Measured on the golden set,
# this pattern neutralizes 7 of the 31 MUST_MISS pairs for 1 of 29 MUST_HIT lost.
# Pass skip_pattern=None to disable, or your own compiled pattern to extend it.
ID_PATTERN = re.compile(r"\b([A-Z]{2,}-\d+|\d{3,}|v?\d+\.\d+)\b")


def _default_embedder(model_name=DEFAULT_MODEL):
    """sentence-transformers, imported and loaded on first use so that importing
    semcache stays fast and tests can inject a stub without the 90MB download.
    """
    model = []

    def embed(text):
        if not model:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            model.append(SentenceTransformer(model_name))
        return np.asarray(model[0].encode(text), dtype="float32")

    return embed


class Cache:
    """A semantic cache for LLM responses.

    :param path: sqlite file, or ":memory:"
    :param embedder: Callable[[str], np.ndarray]. Defaults to sentence-transformers
    :param dim: embedding dimension. Inferred from the first embed if omitted
    :param threshold: cosine similarity required for a hit, [-1, 1]
    :param max_size: live entries before the LRU starts evicting
    :param top_k: candidates pulled from the index per query
    :param skip_pattern: prompts matching it never consult the cache; None disables
    :param max_age: seconds before an entry goes stale; None means never
    """

    def __init__(self, path=":memory:", embedder=None, dim=None,
                 threshold=DEFAULT_THRESHOLD, max_size=1000, top_k=5,
                 skip_pattern=ID_PATTERN, max_age=None, **store_kwargs):
        self._embed_fn = embedder if embedder is not None else _default_embedder()
        self.threshold = threshold
        self.top_k = top_k
        self.skip_pattern = skip_pattern
        self.max_age = max_age
        self._store = Store(str(path), dim=dim, max_size=max_size,
                            max_age=max_age, **store_kwargs)

    def _embed(self, text):
        return np.asarray(self._embed_fn(text), dtype="float32").ravel()

    def get(self, prompt, threshold=None, session_id=None, max_age=None):
        """The best cached answer scoring at or above the threshold, else None.

        A per-call threshold overrides the instance default -- useful for A/B
        calibration against live traffic without standing up a second cache.

        max_age (seconds) overrides the instance default for this call only.

        Returns None without looking if the prompt carries an opaque identifier
        (see ID_PATTERN). A false miss costs one API call; a false hit returns a
        wrong answer to a user, silently. Those are not comparable.
        """
        if self.skip_pattern is not None and self.skip_pattern.search(prompt):
            return None
        limit = self.threshold if threshold is None else threshold
        emb = self._embed(prompt)
        for score, row_id in self._store.search(emb, self.top_k, session_id,
                                                max_age):
            # Written as the positive test, not as `not (score < limit)`: those
            # differ exactly on NaN, which is False for BOTH comparisons and
            # would otherwise fall through and be served as a hit.
            if not score >= limit:
                continue
            row = self._store.fetch(row_id, session_id, max_age)
            if row is None:
                continue               # evicted, compacted, or crash-orphaned --
                                       # a stale index entry, not an error
            return row[1]
        return None

    def put(self, prompt, answer, session_id=None):
        """Store one answer, return its row id.

        No deduplication: storing the same prompt twice creates two rows. Dedupe
        on the caller side if you need it -- in here it is a lookup on every
        write, which is the cost you are trying to avoid.

        Returns None without storing if the prompt carries an opaque identifier.
        The guard has to cover the write too: get() always misses on such a
        prompt, so the caller's get-miss-put loop would store a fresh row on
        every occurrence, and those rows stay in the index as candidates for
        queries that do NOT carry an identifier -- which is the wrong-answer
        case the guard exists to prevent.
        """
        if self.skip_pattern is not None and self.skip_pattern.search(prompt):
            return None
        row_id = self._store.put(prompt, answer, self._embed(prompt), session_id)
        self._store.maybe_compact()
        return row_id

    def search(self, prompt, top_k=None, session_id=None, max_age=None):
        """(score, question, answer) best first, UNFILTERED by threshold.

        This is the calibration tool: it shows you what the threshold is cutting.
        """
        emb = self._embed(prompt)
        out = []
        for score, row_id in self._store.search(emb, top_k or self.top_k,
                                                session_id, max_age):
            row = self._store.fetch(row_id, session_id, max_age)
            if row is not None:
                out.append((score, row[0], row[1]))
        return out

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
