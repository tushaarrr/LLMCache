"""embed -> search -> threshold -> answer.

semcache is not an LLM SDK wrapper. You call your own model; this only decides
whether you need to.
"""

import asyncio
import hashlib
import inspect
import json
import re
import time

import cachetools
import numpy as np

from .store import Store

# GPTCache's default translated. It searches with faiss L2 distance over
# normalized vectors and accepts distance <= 0.8; for normalized vectors
# L2^2 = 2 - 2*cos, so 0.8^2 = 2 - 2*cos gives cos = 0.68.
DEFAULT_THRESHOLD = 0.68
DEFAULT_MODEL = "all-MiniLM-L6-v2"
# NOT ms-marco. That model scores query->passage relevance; the question here is
# "do these two questions want the same answer", which is duplicate-question
# detection. Measured on the golden set, ms-marco reaches zero false hits only at
# zero recall. quora-distilroberta is trained on exactly this task.
DEFAULT_CROSS_ENCODER = "cross-encoder/quora-distilroberta-base"

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
    :param shadow: never serve, only record what would have been served
    :param shadow_log: path for the shadow JSONL
    :param reranker: `(query, [questions]) -> [scores]`; None keeps cosine only
    :param rerank_threshold: score required for a hit when reranking
    :param rerank_top_k: candidates retrieved for the reranker to score
    :param verifier: `(query, question) -> bool`, consulted only inside the band
    :param verify_band: `(lo, hi)` on the ACTIVE scorer's scale -- cosine
        normally, rerank scores when a reranker is set
    """

    def __init__(self, path=":memory:", embedder=None, dim=None,
                 threshold=DEFAULT_THRESHOLD, max_size=1000, top_k=5,
                 skip_pattern=ID_PATTERN, max_age=None, shadow=False,
                 shadow_log=None, now=None, reranker=None,
                 rerank_threshold=None, rerank_top_k=20, verifier=None,
                 verify_band=(0.60, 0.85), **store_kwargs):
        self._embed_fn = embedder if embedder is not None else _default_embedder()
        self.threshold = threshold
        self.top_k = top_k
        self.skip_pattern = skip_pattern
        self.max_age = max_age
        if reranker is not None and rerank_threshold is None:
            raise ValueError(
                "rerank_threshold is required when a reranker is set: "
                "cross-encoder scores are on a different scale from cosine, so "
                "there is no sensible default to inherit from `threshold`"
            )
        self.reranker = reranker
        self.rerank_threshold = rerank_threshold
        self.rerank_top_k = rerank_top_k
        self.verifier = verifier
        self.verify_band = verify_band
        self.verify_stats = {"considered": 0, "above": 0, "below": 0,
                             "verified": 0, "rejected": 0, "errored": 0}
        self.shadow = shadow
        self._shadow_log = str(shadow_log) if shadow_log is not None else None
        self._now = now or time.time
        # L0: exact repeats skip the embedder entirely. Values are row ids, not
        # answers, so every hit still goes through Store.fetch -- which is where
        # the session clause, the state check and the max_age filter live.
        self._l0 = cachetools.LRUCache(max_size)
        self._store = Store(str(path), dim=dim, max_size=max_size,
                            max_age=max_age, now=self._now, **store_kwargs)

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
        if self._skip(prompt):
            return self._finish(prompt, None, session_id)
        limit = self.threshold if threshold is None else threshold
        hit = self._l0_lookup(prompt, session_id, max_age, limit)
        if hit is None:
            emb = self._embed(prompt)
            hit = (self._rerank(prompt, emb, session_id, max_age)
                   if self.reranker is not None
                   else self._lookup(emb, threshold, session_id, max_age, prompt))
        return self._finish(prompt, hit, session_id)

    def _finish(self, prompt, hit, session_id):
        """Serve the answer -- or, in shadow mode, serve nothing and record it."""
        if not self.shadow:
            return hit[2] if hit is not None else None
        record = {
            "prompt": prompt,
            "matched_question": hit[1] if hit else None,
            "score": hit[0] if hit else None,
            "would_hit": hit is not None,
            "session_id": session_id,
            "ts": self._now(),
        }
        if self._shadow_log is not None:
            try:
                with open(self._shadow_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
            except Exception:       # noqa: BLE001 -- a broken log must never
                pass                # break the request path it is measuring
        return None

    def _skip(self, prompt):
        return self.skip_pattern is not None and bool(self.skip_pattern.search(prompt))

    def _l0_key(self, prompt, session_id):
        """Exact-match key. The tenant is part of it, not an afterthought."""
        digest = hashlib.blake2b(prompt.strip().lower().encode("utf-8"),
                                 digest_size=16).digest()
        return (digest, None if session_id is None else str(session_id))

    def _l0_lookup(self, prompt, session_id, max_age, limit):
        """An exact repeat, resolved without embedding anything.

        Returns the candidate tuple, or None to fall through to the semantic
        path. Three things it must not skip:

        - `Store.fetch`, which is where the session clause, the state check and
          the max_age filter live. L0 holds row ids, never answers.
        - the threshold: an exact match scores 1.0, and a caller asking for
          more than that is asking for a miss.
        - a row id that no longer resolves is dropped on the way past, so the
          dict cannot silently fill with dead keys.
        """
        if not 1.0 >= limit:
            return None
        key = self._l0_key(prompt, session_id)
        row_id = self._l0.get(key)
        if row_id is None:
            return None
        row = self._store.fetch(row_id, session_id, max_age)
        if row is None:
            self._l0.pop(key, None)
            return None
        return (1.0, row[0], row[1])

    def _verify(self, prompt, question, score):
        """Band-limited second opinion. True means serve.

        Above the band the score speaks for itself; below it the answer is a
        miss anyway. Only the uncertain middle pays for a call.

        Fails CLOSED: a verifier that raises is treated as a rejection. A miss
        costs one API call, a wrongly-served hit costs a wrong answer, and those
        are not comparable.
        """
        if self.verifier is None:
            return True
        lo, hi = self.verify_band
        self.verify_stats["considered"] += 1
        if score >= hi:
            self.verify_stats["above"] += 1
            return True
        if score < lo:
            self.verify_stats["below"] += 1
            return False
        self.verify_stats["verified"] += 1
        try:
            ok = bool(self.verifier(prompt, question))
        except Exception:       # noqa: BLE001 -- fail closed, never leak upward
            self.verify_stats["errored"] += 1
            return False
        if not ok:
            self.verify_stats["rejected"] += 1
        return ok

    def band_rate(self):
        """Share of considered hits that actually cost a verifier call.

        If most traffic lands in the band, the thresholds are wrong rather than
        the verifier.
        """
        total = self.verify_stats["considered"]
        return self.verify_stats["verified"] / total if total else 0.0

    def _rerank(self, prompt, emb, session_id, max_age):
        """Two-stage retrieval: bi-encoder retrieves, cross-encoder decides.

        A bi-encoder encodes each sentence alone and never sees the pair, which
        is why it cannot tell "celsius to fahrenheit" from its reverse. A
        cross-encoder reads both together, so it can.

        The cosine `threshold` is deliberately NOT applied here. Retrieval
        should be generous and the decision belongs to the reranker; filtering
        on cosine first would discard exactly the pairs the reranker exists to
        judge. The two scores are on different scales and stay separate floats.
        """
        live = []
        for _score, row_id in self._store.search(emb, self.rerank_top_k,
                                                 session_id, max_age):
            row = self._store.fetch(row_id, session_id, max_age)
            if row is not None:
                live.append(row)
        if not live:
            return None
        scores = self.reranker(prompt, [row[0] for row in live])
        best = max(range(len(live)), key=lambda i: scores[i])
        if not scores[best] >= self.rerank_threshold:
            return None
        if not self._verify(prompt, live[best][0], float(scores[best])):
            return None
        return (float(scores[best]), live[best][0], live[best][1])

    def _lookup(self, emb, threshold, session_id, max_age, prompt=None):
        """Best qualifying candidate as (score, question, answer), else None.

        The sync half of get, minus the embedding. Shared with aget, and it
        returns the whole candidate rather than just the answer so shadow mode
        can record what it would have served.
        """
        limit = self.threshold if threshold is None else threshold
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
            if not self._verify(prompt, row[0], score):
                continue           # a rejected candidate does not shadow the
                                   # next one; it just does not get served
            return (score, row[0], row[1])
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
        if self._skip(prompt):
            return None
        return self._write(prompt, answer, self._embed(prompt), session_id)

    def _write(self, prompt, answer, emb, session_id):
        """The sync half of put, minus the embedding. Shared with aput."""
        row_id = self._store.put(prompt, answer, emb, session_id)
        # Keep the FIRST id for a repeated prompt. put deliberately does not
        # dedupe, and the documented rule is that ties resolve to the lower id;
        # overwriting here would quietly make L0 serve the newest instead.
        key = self._l0_key(prompt, session_id)
        if key not in self._l0:
            self._l0[key] = row_id
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

    async def _aembed(self, text):
        """Await a coroutine embedder directly; push a sync one to a thread.

        An API-backed embedder is the common async case, and awaiting it here
        rather than inside to_thread keeps a worker thread from sitting blocked
        on network I/O the event loop could have been doing.
        """
        if inspect.iscoroutinefunction(self._embed_fn):
            return np.asarray(await self._embed_fn(text), dtype="float32").ravel()
        return await asyncio.to_thread(self._embed, text)

    async def aget(self, prompt, threshold=None, session_id=None, max_age=None):
        """`get` without blocking the event loop. Same semantics."""
        if self._skip(prompt):
            return self._finish(prompt, None, session_id)
        limit = self.threshold if threshold is None else threshold
        hit = self._l0_lookup(prompt, session_id, max_age, limit)
        if hit is not None:
            return self._finish(prompt, hit, session_id)
        emb = await self._aembed(prompt)
        if self.reranker is not None:
            hit = await asyncio.to_thread(self._rerank, prompt, emb, session_id,
                                          max_age)
        else:
            hit = await asyncio.to_thread(self._lookup, emb, threshold,
                                          session_id, max_age, prompt)
        return self._finish(prompt, hit, session_id)

    async def aput(self, prompt, answer, session_id=None):
        """`put` without blocking the event loop. Same semantics."""
        if self._skip(prompt):
            return None
        emb = await self._aembed(prompt)
        return await asyncio.to_thread(self._write, prompt, answer, emb, session_id)

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def cross_encoder_reranker(model_name=DEFAULT_CROSS_ENCODER):
    """The shipped reranker. Loaded on first use, like the default embedder.

    Returns `(query, [questions]) -> [scores]`. Scores are logits, not cosines --
    calibrate rerank_threshold separately.
    """
    model = []

    def rerank(query, candidates):
        if not model:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            model.append(CrossEncoder(model_name))
        return list(model[0].predict([(query, c) for c in candidates]))

    return rerank


def shadow_report(path):
    """Summarize a shadow log: what the cache would have served, and how often.

    Returns a dict; print it or assert on it. The point of shadow mode is that
    this runs on real traffic instead of on hand-written pairs.
    """
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    hits = [r for r in records if r["would_hit"]]
    scores = sorted((r["score"] for r in hits), reverse=True)
    buckets = {}
    for score in scores:
        edge = min(int(score * 10) / 10, 0.9)
        buckets[round(edge, 1)] = buckets.get(round(edge, 1), 0) + 1

    return {
        "total": len(records),
        "would_hit": len(hits),
        "would_hit_rate": len(hits) / len(records) if records else 0.0,
        "score_distribution": dict(sorted(buckets.items(), reverse=True)),
        "top_20": [
            {"prompt": r["prompt"], "matched_question": r["matched_question"],
             "score": r["score"]}
            for r in sorted(hits, key=lambda r: -r["score"])[:20]
        ],
    }
