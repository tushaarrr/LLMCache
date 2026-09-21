"""L0: exact repeats skip the embedder.

The embedder is 70-85% of a hit. An exact repeat should not pay a model forward
pass to rediscover what a dict already knows -- but it must not skip any of the
checks that make a hit safe.
"""

import numpy as np
import pytest

from semcache import Cache

from .conftest import DIM, stub


class Counting:
    def __init__(self, fn=stub):
        self.fn, self.calls = fn, 0

    def __call__(self, text):
        self.calls += 1
        return self.fn(text)


@pytest.fixture
def counting():
    return Counting()


@pytest.fixture
def c(counting):
    cache = Cache(":memory:", embedder=counting, dim=DIM)
    yield cache
    cache.close()


def test_l0_hit_skips_the_embedder(c, counting):
    c.put("what is git", "vcs")
    n = counting.calls
    assert c.get("what is git") == "vcs"
    assert counting.calls == n, "an exact repeat still paid for an embedding"


def test_l0_is_case_and_whitespace_insensitive(c, counting):
    c.put("What Is Git", "vcs")
    n = counting.calls
    assert c.get("  what is git  ") == "vcs"
    assert counting.calls == n


# -- the three traps ---------------------------------------------------------
def test_l0_respects_session(c):
    """A prompt-only key would let tenant B hit tenant A's row with no scoring
    involved at all -- F10 reintroduced at a new layer.
    """
    c.put("q", "a", session_id="acme")
    assert c.get("q", session_id="globex") is None


def test_l0_keys_are_per_tenant(c, counting):
    """The discriminating version of the test above.

    With the session left out of the key, the second put overwrites the first,
    so acme's lookup misses L0 and falls through to the embedder. Store.fetch
    keeps that *safe*, which is why the miss -- not a leak -- is what to assert.
    """
    c.put("q", "acme answer", session_id="acme")
    c.put("q", "globex answer", session_id="globex")
    n = counting.calls
    assert c.get("q", session_id="acme") == "acme answer"
    assert c.get("q", session_id="globex") == "globex answer"
    assert counting.calls == n, "a tenant's L0 entry was clobbered by another's"


def test_l0_respects_max_age(c, clock):
    cache = Cache(":memory:", embedder=stub, dim=DIM, now=clock)
    cache.put("q", "a")
    clock.advance(3600)
    assert cache.get("q", max_age=60) is None
    cache.close()


def test_l0_respects_soft_delete(c):
    row_id = c.put("q", "a")
    c._store.soft_delete([row_id])
    assert c.get("q") is None


def test_l0_drops_dead_keys_on_the_way_past(c):
    """Stale ids are safe (AUTOINCREMENT never reuses them) but they must not
    accumulate and quietly shrink the L0 hit rate.
    """
    row_id = c.put("q", "a")
    assert len(c._l0) == 1
    c._store.soft_delete([row_id])
    c.get("q")
    assert len(c._l0) == 0


def test_l0_respects_state_after_compaction(c):
    row_id = c.put("q", "a")
    c._store.soft_delete([row_id])
    c._store.compact()
    assert c.get("q") is None


# -- L0 must not bypass the scoring contract ---------------------------------
def test_l0_respects_the_threshold(c):
    """An exact match scores 1.0; a caller asking for more wants a miss."""
    c.put("q", "a")
    assert c.get("q", threshold=1.01) is None
    assert c.get("q") == "a"


def test_l0_keeps_the_lower_id_on_duplicates(c):
    """put does not dedupe and ties resolve to the lower id. Overwriting the L0
    entry on the second put would quietly serve the newest instead.
    """
    c.put("q", "first")
    c.put("q", "second")
    assert c.get("q") == "first"


def test_l0_is_bounded():
    cache = Cache(":memory:", embedder=stub, dim=DIM, max_size=10)
    for i in range(50):
        cache.put(f"q{i}", f"a{i}")
    assert len(cache._l0) <= 10
    cache.close()


def test_l0_honours_shadow_mode(tmp_path):
    import json
    log = tmp_path / "s.jsonl"
    cache = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
                  shadow_log=log)
    cache.put("q", "a")
    assert cache.get("q") is None             # L0 hit must not serve in shadow
    rec = json.loads(open(log).readline())
    assert rec["would_hit"] is True and rec["score"] == 1.0
    cache.close()


def test_l0_works_on_the_async_path():
    import asyncio
    counting = Counting()
    cache = Cache(":memory:", embedder=counting, dim=DIM)

    async def main():
        await cache.aput("q", "a")
        n = counting.calls
        got = await cache.aget("q")
        return got, n, counting.calls

    got, before, after = asyncio.run(main())
    assert got == "a" and before == after
    cache.close()
