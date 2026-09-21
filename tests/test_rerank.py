"""Two-stage retrieval: bi-encoder retrieves, cross-encoder decides.

Needs the real cross-encoder, so everything that scores text is marked slow.

WHAT THIS STEP ACHIEVED, MEASURED
---------------------------------
Threshold picked on the CALIBRATION half only (even indices), by the rule
"highest MUST_MISS score there, plus 0.005" = 0.0322. Holdout (odd indices) was
not consulted in choosing it. ID guard applied to both arms.

    arm                              recall   false hits
    cosine >= 0.68   [calibration]     93%       7/11
    rerank >= 0.0322 [calibration]     93%       0/11
    cosine >= 0.68   [HOLDOUT]         93%      12/13
    rerank >= 0.0322 [HOLDOUT]         93%       7/13

**At identical recall, wrong answers on the holdout half drop from 12 to 7.**
The calibration half goes to zero by construction; the holdout keeping 7 is the
generalization gap, and it is the number that counts.

It did NOT reach the step's target of zero false hits on holdout, and no
threshold does: raising it to 0.5 leaves 3 while dropping recall to 71%. Three
of the survivors score 0.96+, so no threshold can separate them at any useful
recall. That is why Step 4's gate opens.

Orientation matters and is easy to get wrong: cross-encoders are not symmetric,
and the cache calls reranker(query, [stored_question]). Calibrating on
(a, [b]) produces a threshold the running cache never uses -- it put the
negation pair at 0.027 when the cache actually sees 0.054.
"""

import numpy as np
import pytest

from semcache import Cache, cross_encoder_reranker

from .conftest import DIM, stub

# Chosen on the calibration half: the lowest threshold with zero false hits there.
RERANK_THRESHOLD = 0.0322


@pytest.fixture(scope="module")
def reranker():
    pytest.importorskip("sentence_transformers")
    return cross_encoder_reranker()


@pytest.fixture
def reranking_cache(reranker):
    c = Cache(":memory:", reranker=reranker, rerank_threshold=RERANK_THRESHOLD,
              skip_pattern=None)
    yield c
    c.close()


# -- config ------------------------------------------------------------------
def test_reranker_requires_an_explicit_threshold():
    """Cross-encoder logits are not cosines. One number meaning two things is
    how the original ended up with nine evaluators and a range() indirection.
    """
    with pytest.raises(ValueError, match="rerank_threshold"):
        Cache(":memory:", embedder=stub, dim=DIM, reranker=lambda q, c: [1.0])


def test_rerank_off_by_default_changes_nothing(cache):
    assert cache.reranker is None
    cache.put("what is github?", "a code host")
    assert cache.get("what is github?") == "a code host"
    assert cache.get("how do I cook risotto") is None


def test_reranker_is_a_plain_callable():
    """No registry, no ABC -- a callable and a float."""
    calls = []

    def fake(query, candidates):
        calls.append((query, tuple(candidates)))
        return [9.0] * len(candidates)

    c = Cache(":memory:", embedder=stub, dim=DIM, reranker=fake,
              rerank_threshold=1.0, skip_pattern=None)
    c.put("q", "a")
    assert c.get("different prompt") == "a"     # fake accepts everything
    assert calls and calls[0][1] == ("q",)
    c.close()


def test_rerank_rejection_is_a_miss():
    c = Cache(":memory:", embedder=stub, dim=DIM,
              reranker=lambda q, cand: [-99.0] * len(cand),
              rerank_threshold=0.0, skip_pattern=None)
    c.put("q", "a")
    assert c.get("q2") is None
    c.close()


def test_rerank_still_honours_sessions_and_ttl(clock):
    c = Cache(":memory:", embedder=stub, dim=DIM,
              reranker=lambda q, cand: [9.0] * len(cand),
              rerank_threshold=0.0, skip_pattern=None, now=clock)
    c.put("q", "a", session_id="acme")
    assert c.get("qq", session_id="globex") is None      # rerank never sees it
    assert c.get("qq", session_id="acme") == "a"
    clock.advance(3600)
    assert c.get("qq", session_id="acme", max_age=60) is None
    c.close()


# -- the measured behaviour ---------------------------------------------------
@pytest.mark.slow
def test_rerank_beats_cosine_at_equal_recall(reranker):
    """The actual win: same recall, fewer wrong answers, measured on holdout."""
    import numpy as np

    from semcache.cache import ID_PATTERN

    from .golden_pairs import MUST_HIT, MUST_MISS, default_embedder, split

    embed = default_embedder()
    memo = {}

    def E(t):
        if t not in memo:
            memo[t] = embed(t)
        return memo[t]

    guard = lambda p: bool(ID_PATTERN.search(p.a) or ID_PATTERN.search(p.b))
    _, hold_hit = split(MUST_HIT)
    _, hold_miss = split(MUST_MISS)
    hits = [p for p in hold_hit if not guard(p)]
    misses = [p for p in hold_miss if not guard(p)]

    cos_recall = sum(1 for p in hits if np.dot(E(p.a), E(p.b)) >= 0.68) / len(hits)
    cos_false = sum(1 for p in misses if np.dot(E(p.a), E(p.b)) >= 0.68)
    ce_recall = sum(1 for p in hits
                    if reranker(p.b, [p.a])[0] >= RERANK_THRESHOLD) / len(hits)
    ce_false = sum(1 for p in misses
                   if reranker(p.b, [p.a])[0] >= RERANK_THRESHOLD)

    assert ce_recall >= cos_recall - 0.05, "rerank bought safety with recall"
    assert ce_false < cos_false, f"rerank did not reduce false hits ({ce_false} vs {cos_false})"
    assert ce_false <= 7, f"surviving false hits grew to {ce_false}"


@pytest.mark.slow
def test_rerank_known_surviving_false_hits(reranking_cache):
    """The seven the cross-encoder does NOT catch on the holdout half.

    Pinned as a known-bad baseline, not as desired behaviour. If one starts
    being caught this test fails and the list should shrink -- that is the
    point. Step 4 (verify-on-hit) exists because this list is not empty.

    The first three score 0.96+: no threshold separates them at any useful
    recall, so they are not a tuning problem.
    """
    surviving = [
        ("what is a P0 incident", "what is a P1 incident"),                      # 0.986
        ("convert celsius to fahrenheit", "convert fahrenheit to celsius"),      # 0.982
        ("how do I delete a file in bash", "how do I delete a directory in bash"),  # 0.962
        ("explain HTTP", "explain HTTPS"),                                       # 0.071
        ("is python garbage collected", "is python not garbage collected"),      # 0.054
        ("how many miles in a marathon", "how many kilometers in a marathon"),   # 0.051
        ("translate this to french", "translate this from french"),              # 0.035
    ]
    still_wrong = [(a, b) for a, b in surviving
                   if _round_trip(reranking_cache, a, b)]
    assert len(still_wrong) == 7, (
        f"the surviving-false-hit set changed ({len(still_wrong)}/7 still wrong). "
        "If it shrank, update this list and the numbers in the module docstring."
    )


def _round_trip(cache, a, b):
    cache.put(a, "X")
    return cache.get(b) == "X"


@pytest.mark.slow
def test_rerank_preserves_must_hit_recall(reranker):
    """Recall on the holdout half must not fall below the measured 79% - 10."""
    from .golden_pairs import MUST_HIT, split
    from semcache.cache import ID_PATTERN

    _, holdout = split(MUST_HIT)
    live = [p for p in holdout
            if not (ID_PATTERN.search(p.a) or ID_PATTERN.search(p.b))]
    kept = sum(1 for p in live
               if float(reranker(p.b, [p.a])[0]) >= RERANK_THRESHOLD)
    recall = kept / len(live)
    assert recall >= 0.83, f"rerank recall regressed to {recall:.0%}"


@pytest.mark.slow
def test_rerank_latency_budget(reranker):
    """20 candidates scored in < 40ms p50 on CPU."""
    import time

    cands = [f"what is candidate question number {i}" for i in range(20)]
    reranker("warm up the model", cands)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        reranker("what is the capital of France", cands)
        times.append(time.perf_counter() - t)
    p50 = sorted(times)[len(times) // 2]
    assert p50 < 0.040, f"rerank p50 {p50 * 1000:.0f}ms exceeds the 40ms budget"


def test_rerank_must_go_through_fetch_not_the_raw_row():
    """Session and TTL are masked in the index before the top-k cut, so they
    survive a fetch bypass. `state = 0` does NOT: soft-deleted rows stay in the
    index on purpose, and fetch is the only thing that filters them. A rerank
    path that reads rows directly serves evicted answers.
    """
    c = Cache(":memory:", embedder=stub, dim=DIM,
              reranker=lambda q, cand: [9.0] * len(cand),
              rerank_threshold=0.0, skip_pattern=None)
    row_id = c.put("q", "a")
    c._store.soft_delete([row_id])
    assert c.get("q2") is None, "rerank served a soft-deleted row"
    c.close()


def test_rerank_applies_on_the_async_path():
    import asyncio

    seen = []

    def fake(query, candidates):
        seen.append(query)
        return [-99.0] * len(candidates)      # reject everything

    c = Cache(":memory:", embedder=stub, dim=DIM, reranker=fake,
              rerank_threshold=0.0, skip_pattern=None)

    async def main():
        await c.aput("q", "a")
        return await c.aget("q2")

    assert asyncio.run(main()) is None, "aget skipped the reranker"
    assert seen == ["q2"], "the reranker was never consulted on the async path"
    c.close()
