"""F9: staleness. Nothing expired before this; "what is the latest Python
version" was served forever.

All clock-driven -- no test here sleeps.
"""

import numpy as np
import pytest

from semcache import Cache
from semcache.store import Store

from .conftest import DIM, at_cosine, stub


def test_expired_entry_is_a_miss(tmp_path, clock):
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, now=clock)
    c.put("q", "a")
    assert c.get("q", max_age=60) == "a"
    clock.advance(3600)
    assert c.get("q", max_age=60) is None
    c.close()


def test_max_age_none_never_expires(tmp_path, clock):
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, now=clock)
    c.put("q", "a")
    clock.advance(10 ** 9)
    assert c.get("q", max_age=None) == "a"     # ...so nothing expires
    c.close()


def test_instance_max_age_applies_and_per_call_overrides(tmp_path, clock):
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, now=clock, max_age=60)
    c.put("q", "a")
    clock.advance(3600)
    assert c.get("q") is None                  # instance default expires it
    assert c.get("q", max_age=10 ** 6) == "a"  # per-call override widens
    c.close()


def test_expired_row_does_not_consume_a_topk_slot(tmp_path, clock):
    """The expired row outscores every fresh one. If expiry were applied after
    the top-k cut it would eat a slot and hide a live entry behind it.
    """
    hi, lo = at_cosine(0.99)
    embs = {"stale": hi, "query": hi}
    for i in range(5):
        embs[f"fresh{i}"] = lo
    c = Cache(tmp_path / "c.db", embedder=embs.__getitem__, dim=DIM, top_k=5,
              now=clock, threshold=0.0)
    c.put("stale", "STALE ANSWER")
    clock.advance(3600)
    for i in range(5):
        c.put(f"fresh{i}", f"fresh answer {i}")

    got = c.get("query", max_age=60)
    assert got is not None, "all five fresh rows were hidden behind one stale one"
    assert got != "STALE ANSWER"
    assert len(c.search("query", top_k=5, max_age=60)) == 5
    c.close()


def test_expired_rows_are_compacted(tmp_path, clock):
    """Expired entries must eventually leave the file, not just be hidden."""
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, now=clock, max_age=60)
    for i in range(20):
        c.put(f"q{i}", f"a{i}")
    assert c._store.count(all=True) == 20
    clock.advance(3600)
    c.put("fresh", "fresh")                    # put -> maybe_compact -> sweep
    assert c._store.count(state=0) == 1
    assert c._store.count(all=True) == 1, "stale rows never left the file"
    c.close()


def test_sweep_is_a_noop_without_a_default_max_age(tmp_path, clock):
    """A per-call max_age hides rows; only the instance default reclaims them."""
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, now=clock)
    for i in range(20):
        c.put(f"q{i}", f"a{i}")
    clock.advance(3600)
    c.put("fresh", "fresh")
    assert c._store.count(all=True) == 21
    c.close()


def test_expiry_survives_restart(tmp_path, clock):
    path = tmp_path / "c.db"
    c = Cache(path, embedder=stub, dim=DIM, now=clock)
    c.put("q", "a")
    c.close()
    clock.advance(3600)
    c = Cache(path, embedder=stub, dim=DIM, now=clock)
    assert c.get("q", max_age=60) is None
    assert c.get("q", max_age=10 ** 6) == "a"
    c.close()


def test_store_level_cutoff_is_inclusive(clock):
    s = Store(":memory:", dim=DIM, now=clock)
    row = s.put("q", "a", stub("q"))
    clock.advance(60)
    assert s.fetch(row, max_age=60) is not None    # exactly on the boundary
    clock.advance(0.001)
    assert s.fetch(row, max_age=60) is None
    s.close()
