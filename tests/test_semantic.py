"""Layer 2: the semantic contract, and its regression gate.

Only this layer needs the real embedder, because only this layer is testing the
embedder.

    python -m tests.golden_pairs      # recalibrate

READ THIS BEFORE TRUSTING A THRESHOLD
-------------------------------------
A plausible-looking sweep would suggest 0.75 is "safe" with zero false hits.
Measured against all-MiniLM-L6-v2 on this dataset, it is not true, and nothing
near it is:

    thresh  recall  false hits
     0.68     93%      25
     0.75     90%      22
     0.90     66%      12
     1.00      7%       0

No threshold separates the sets. Three families are irreducible -- direction
("celsius to fahrenheit" / "fahrenheit to celsius", 0.994), negation ("is python
garbage collected" / "is python NOT garbage collected", 0.981) and unit ("30
seconds" / "30 minutes", 0.970). Each is one token against a near-identical
sentence, which is exactly what a bag-of-meaning embedder cannot see.

So the gate below is a REGRESSION gate, not a safety proof. It freezes the
measured baseline and fails when it gets worse. What actually reduces the danger
is Cache's ID_PATTERN guard (7 of 31 neutralized for 1 of 29 lost) plus, for the
rest, a domain embedder or a cross-encoder rerank -- not a bigger float.
"""

import numpy as np
import pytest

from semcache.cache import DEFAULT_THRESHOLD, ID_PATTERN

from .golden_pairs import MUST_HIT, MUST_MISS, _self_check, default_embedder

# The gate measures the configuration that actually ships.
PRODUCTION_THRESHOLD = DEFAULT_THRESHOLD

# Frozen 2026-09-21, all-MiniLM-L6-v2, post-ID-guard, at 0.68. These already
# serve wrong answers. The gate's job is to stop the list GROWING.
KNOWN_FALSE_HITS = {
    "convert celsius to fahrenheit",              # 0.994 direction
    "how to migrate from mysql to postgres",      # 0.990 direction
    "is python garbage collected",                # 0.981 negation
    "set the timeout to 30 seconds",              # 0.970 unit
    "translate this to french",                   # 0.962 direction
    "why should I use microservices",             # 0.951 negation
    "what changed in postgres 14",                # 0.934 version
    "python 2 print syntax",                      # 0.891 version
    "how do I enable swap on linux",              # 0.877 negation
    "how many miles in a marathon",               # 0.876 unit
    "explain the react useEffect hook",           # 0.852 entity swap
    "how do I upgrade to node 18",                # 0.849 version
    "what is a P0 incident",                      # 0.831 internal severity
    "give me 3 examples",                         # 0.812 magnitude
    "what is the weather today",                  # 0.808 temporal
    "how do I delete a file in bash",             # 0.795 scope
    "what is the population of Tokyo",            # 0.786 entity swap
    "how to make a container writable",           # 0.736 antonym
    "explain HTTP",                               # 0.693 protocol
}

BASELINE_RECALL = 0.93


def _guarded(pair):
    return bool(ID_PATTERN.search(pair.a) or ID_PATTERN.search(pair.b))


# -- always on: no model required --------------------------------------------
def test_dataset_has_not_rotted():
    _self_check()


def test_identifier_guard_covers_the_opaque_id_families():
    """The guard is ours, so it is testable without the model -- and it is the
    only defense that works on these families at all.
    """
    caught = {p.why for p in MUST_MISS if _guarded(p)}
    assert {"ticket id", "sku", "cve id", "status code"} <= caught


def test_identifier_guard_is_cheap_on_recall():
    assert sum(_guarded(p) for p in MUST_HIT) <= 2


def test_known_false_hits_are_all_real_pairs():
    """Guards the gate itself: a renamed prompt must not silently empty the set."""
    prompts = {p.a for p in MUST_MISS}
    assert KNOWN_FALSE_HITS <= prompts


# -- slow: needs the real embedder --------------------------------------------
@pytest.fixture(scope="module")
def scored():
    pytest.importorskip("sentence_transformers")
    embed = default_embedder()
    memo = {}

    def emb(t):
        if t not in memo:
            memo[t] = embed(t)
        return memo[t]

    def sim(p):
        return float(np.dot(emb(p.a), emb(p.b)))

    live_hit = [(p, sim(p)) for p in MUST_HIT if not _guarded(p)]
    live_miss = [(p, sim(p)) for p in MUST_MISS if not _guarded(p)]
    return live_hit, live_miss


@pytest.mark.slow
def test_no_new_false_hits(scored):
    """FAIL THE BUILD. Every entry here is a wrong answer served to a user."""
    _, misses = scored
    new = {
        p.a for p, s in misses
        if s >= PRODUCTION_THRESHOLD and p.a not in KNOWN_FALSE_HITS
    }
    assert not new, f"NEW FALSE HITS at {PRODUCTION_THRESHOLD}: {sorted(new)}"


@pytest.mark.slow
def test_known_false_hits_have_not_multiplied(scored):
    _, misses = scored
    count = sum(1 for _, s in misses if s >= PRODUCTION_THRESHOLD)
    assert count <= len(KNOWN_FALSE_HITS)


@pytest.mark.slow
def test_recall_has_not_regressed(scored):
    """Warn-level in spirit: a false miss costs one API call, nothing more."""
    hits, _ = scored
    recall = sum(1 for _, s in hits if s >= PRODUCTION_THRESHOLD) / len(hits)
    assert recall >= BASELINE_RECALL - 0.10, f"recall fell to {recall:.0%}"
