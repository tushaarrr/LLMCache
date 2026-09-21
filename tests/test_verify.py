"""Verify-on-hit: a cheap second opinion, only where the score is uncertain.

Built because Step 3's cross-encoder left 7 false hits on the holdout half.

WHAT THE MEASUREMENT SAYS ABOUT THE BAND
----------------------------------------
The band design assumes danger lives in the uncertain middle. On this dataset
it does not -- see test_verify_band_premise_vs_measured_data. Three of the
seven survivors score 0.96+, above any sane upper bound, so they are served
without ever reaching the verifier. Catching them means hi > 0.99, i.e.
verifying every hit, which is a cost decision rather than a default.
"""

import pytest

from semcache import Cache

from .conftest import DIM, stub


class CountingVerifier:
    def __init__(self, verdict=True):
        self.calls, self.verdict = [], verdict

    def __call__(self, query, question):
        self.calls.append((query, question))
        return self.verdict


def _cache(**kw):
    kw.setdefault("embedder", stub)
    kw.setdefault("dim", DIM)
    kw.setdefault("skip_pattern", None)
    return Cache(":memory:", **kw)


def test_no_verifier_changes_nothing():
    c = _cache()
    c.put("q", "a")
    assert c.get("q") == "a"
    assert c.verify_stats["considered"] == 0
    c.close()


def test_verifier_only_called_in_band():
    """Above the band: served, no call. Below: missed, no call."""
    v = CountingVerifier()
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.99] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") == "a"                  # 0.99 >= hi -> serve unverified
    assert v.calls == []
    assert c.verify_stats["above"] == 1
    c.close()

    v = CountingVerifier()
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.10] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") is None                 # 0.10 < lo -> miss unverified
    assert v.calls == []
    assert c.verify_stats["below"] == 1
    c.close()

    v = CountingVerifier()
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.70] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") == "a"                  # in band -> verified, accepted
    assert v.calls == [("q2", "q")]
    assert c.verify_stats["verified"] == 1
    c.close()


def test_verifier_rejection_is_a_miss():
    v = CountingVerifier(verdict=False)
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.70] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") is None
    assert c.verify_stats["rejected"] == 1
    c.close()


def test_verifier_error_fails_closed():
    """A verifier that raises must produce a miss, not an exception and not a
    served answer. A miss costs one API call; a wrong hit costs a wrong answer.
    """
    def boom(query, question):
        raise RuntimeError("provider down")

    c = _cache(verifier=boom, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.70] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") is None
    assert c.verify_stats["errored"] == 1
    c.close()


def test_verifier_timeout_fails_closed():
    def slow(query, question):
        raise TimeoutError("verifier timed out")

    c = _cache(verifier=slow, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.70] * len(cand), rerank_threshold=0.0)
    c.put("q", "a")
    assert c.get("q2") is None
    c.close()


def test_band_rate_is_reported():
    v = CountingVerifier()
    scores = iter([0.99, 0.70, 0.10, 0.70])
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [next(scores)] * len(cand),
               rerank_threshold=0.0)
    for i in range(4):
        c.put(f"p{i}", "a")
    for i in range(4):
        c.get(f"query{i}")
    assert c.verify_stats["considered"] == 4
    assert c.band_rate() == pytest.approx(0.5)      # 2 of 4 paid for a call
    c.close()


def test_verifier_applies_without_a_reranker():
    """The band runs on cosine when no reranker is configured.

    Controlled vectors so the cosine is deterministic and lands in the band --
    the random stub would make this flaky.
    """
    from .conftest import at_cosine

    a, b = at_cosine(0.75)
    v = CountingVerifier(verdict=False)
    c = Cache(":memory:", embedder={"stored": a, "query": b}.__getitem__,
              dim=DIM, skip_pattern=None, verifier=v, verify_band=(0.60, 0.85))
    c.put("stored", "answer")
    assert c.get("query") is None, "cosine-path verifier rejection was ignored"
    assert v.calls == [("query", "stored")]
    assert c.verify_stats["verified"] == 1

    accept = CountingVerifier(verdict=True)
    c2 = Cache(":memory:", embedder={"stored": a, "query": b}.__getitem__,
               dim=DIM, skip_pattern=None, verifier=accept,
               verify_band=(0.60, 0.85))
    c2.put("stored", "answer")
    assert c2.get("query") == "answer"
    c.close(); c2.close()


def test_rejected_candidate_does_not_shadow_the_next():
    """A rejection is not a short circuit: the loop keeps looking."""
    seen = []

    def only_accept_second(query, question):
        seen.append(question)
        return question == "second"

    c = Cache(":memory:", embedder={"first": stub("x"), "second": stub("x"),
                                    "query": stub("x")}.__getitem__,
              dim=DIM, skip_pattern=None, verifier=only_accept_second,
              verify_band=(0.0, 1.01), threshold=0.0)
    c.put("first", "wrong answer")
    c.put("second", "right answer")
    assert c.get("query") == "right answer"
    assert seen == ["first", "second"]
    c.close()


def test_verifier_applies_on_the_async_path():
    import asyncio

    v = CountingVerifier(verdict=False)
    c = _cache(verifier=v, verify_band=(0.60, 0.85),
               reranker=lambda q, cand: [0.70] * len(cand), rerank_threshold=0.0)

    async def main():
        await c.aput("q", "a")
        return await c.aget("q2")

    assert asyncio.run(main()) is None
    assert v.calls == [("q2", "q")]
    c.close()


@pytest.mark.slow
def test_verify_band_premise_vs_measured_data():
    """The band assumes danger is in the middle. Measured, it is at the top.

    Of the 7 false hits the cross-encoder leaves, 3 score 0.96+ and would be
    served without verification under any band whose upper bound is below that.
    This test documents the gap rather than asserting a fix.
    """
    pytest.importorskip("sentence_transformers")
    from semcache import cross_encoder_reranker

    rr = cross_encoder_reranker()
    survivors = {
        "what is a P0 incident": "what is a P1 incident",
        "convert celsius to fahrenheit": "convert fahrenheit to celsius",
        "how do I delete a file in bash": "how do I delete a directory in bash",
        "explain HTTP": "explain HTTPS",
        "is python garbage collected": "is python not garbage collected",
        "how many miles in a marathon": "how many kilometers in a marathon",
        "translate this to french": "translate this from french",
    }
    scores = {a: float(rr(b, [a])[0]) for a, b in survivors.items()}
    above_default_band = [a for a, s in scores.items() if s >= 0.85]
    assert len(above_default_band) == 3, (
        f"the high-confidence survivor set changed: {above_default_band}"
    )
