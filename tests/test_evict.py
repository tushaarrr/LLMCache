"""The two eviction tiers: LRU soft delete (cheap, frequent) and compaction
(expensive, rare).
"""

import pytest

from semcache import Cache
from semcache.evict import LRU, should_compact
from semcache.store import Store

from .conftest import DIM, stub, vec


# -- I7 ---------------------------------------------------------------------
def test_lru_bounds_live_set(store_maxsize_10):
    """max_size bounds the LIVE set. It does not bound the sqlite file, which
    grows until compaction -- asserting file size would flake.
    """
    for i in range(50):
        store_maxsize_10.put(f"q{i}", f"a{i}", vec(i))
    assert store_maxsize_10.count(state=0) <= 10


def test_eviction_marks_rather_than_deletes(store_maxsize_10):
    for i in range(30):
        store_maxsize_10.put(f"q{i}", f"a{i}", vec(i))
    assert store_maxsize_10.count(state=0) == 10
    assert store_maxsize_10.count(state=-1) == 20
    assert store_maxsize_10.count(all=True) == 30


def test_lru_evicts_least_recently_used(store_maxsize_10):
    ids = [store_maxsize_10.put(f"q{i}", f"a{i}", vec(i)) for i in range(10)]
    store_maxsize_10.fetch(ids[0])                      # touch the oldest
    store_maxsize_10.put("q10", "a10", vec(10))         # forces one eviction
    assert store_maxsize_10.fetch(ids[0]) is not None   # survived, was touched
    assert store_maxsize_10.fetch(ids[1]) is None       # evicted instead


def test_fetch_touches_recency(store_maxsize_10):
    ids = [store_maxsize_10.put(f"q{i}", f"a{i}", vec(i)) for i in range(10)]
    for _ in range(5):
        for row_id in ids[:5]:
            store_maxsize_10.fetch(row_id)
    for i in range(10, 15):
        store_maxsize_10.put(f"q{i}", f"a{i}", vec(i))
    assert all(store_maxsize_10.fetch(r) is not None for r in ids[:5])


def test_evicted_entries_stop_hitting():
    c = Cache(":memory:", embedder=stub, dim=DIM, max_size=5)
    c.put("q0", "a0")
    for i in range(1, 10):
        c.put(f"q{i}", f"a{i}")
    assert c.get("q0") is None
    assert c.get("q9") == "a9"
    c.close()


# -- compaction policy --------------------------------------------------------
def test_should_compact_thresholds():
    assert should_compact(5000, 1_000_000)          # count trigger
    assert not should_compact(4999, 1_000_000)
    assert should_compact(10, 100)                  # rate trigger, exactly 10%
    assert not should_compact(9, 100)
    assert not should_compact(0, 0)
    assert not should_compact(0, 100)


def test_should_compact_defaults_match_the_original():
    from semcache.evict import MAX_MARK_COUNT, MAX_MARK_RATE

    assert MAX_MARK_COUNT == 5000       # EvictionManager.MAX_MARK_COUNT
    assert MAX_MARK_RATE == 0.1         # EvictionManager.MAX_MARK_RATE


def test_maybe_compact_is_a_noop_below_threshold(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(100)]
    store.soft_delete(ids[:5])                      # 5% marked
    assert store.maybe_compact() is False
    assert store.count(all=True) == 100


def test_maybe_compact_fires_above_rate(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(100)]
    store.soft_delete(ids[:15])                     # 15% marked
    assert store.maybe_compact() is True
    assert store.count(all=True) == 85


def test_put_triggers_compaction_and_keeps_the_cache_usable():
    """Compaction runs synchronously from put; the entries that survive it must
    still hit afterwards.
    """
    c = Cache(":memory:", embedder=stub, dim=DIM, max_size=20)
    for i in range(400):
        c.put(f"q{i}", f"a{i}")
    assert c._store.count(state=0) <= 20
    assert c._store.count(all=True) < 400            # compaction reclaimed rows
    assert c.get("q399") == "a399"


def test_compaction_does_not_strand_the_lru(store_maxsize_10):
    """After a rebuild the LRU must still bound the live set."""
    for i in range(60):
        store_maxsize_10.put(f"q{i}", f"a{i}", vec(i))
    store_maxsize_10.compact()
    for i in range(60, 90):
        store_maxsize_10.put(f"q{i}", f"a{i}", vec(i))
    assert store_maxsize_10.count(state=0) <= 10


def test_reload_over_maxsize_rebounds_the_live_set(tmp_path):
    """Reopening with a smaller max_size must re-bound, not blow past it."""
    path = str(tmp_path / "c.db")
    s = Store(path, dim=DIM, max_size=1000)
    for i in range(50):
        s.put(f"q{i}", f"a{i}", vec(i))
    s.close()

    s = Store(path, dim=DIM, max_size=10)
    assert s.count(state=0) <= 10
    s.close()


# -- the LRU hook itself -------------------------------------------------------
def test_lru_calls_on_evict_with_the_popped_key():
    evicted = []
    lru = LRU(2, evicted.append)
    for i in range(4):
        lru[i] = True
    assert evicted == [[0], [1]]


def test_clear_bypasses_the_eviction_hook():
    """Documents WHY the store never calls .clear(): cachetools empties the dict
    directly, so on_evict never fires and the rows would be stranded live and
    unevictable rather than soft-deleted.
    """
    evicted = []
    lru = LRU(10, evicted.append)
    for i in range(5):
        lru[i] = True
    lru.clear()
    assert evicted == [], "if this ever fires, the hazard has inverted"
    assert len(lru) == 0


def test_compaction_preserves_lru_recency():
    """compact() calls _load(). Rebuilding the LRU there from `ORDER BY id`
    would replace recency with insertion order, so the hottest entry would be
    evicted first after every compaction (F3, 'hits vanish after a while').
    """
    s = Store(":memory:", dim=DIM, max_size=3)
    hot = s.put("HOT", "a", vec(0))
    cold = s.put("cold", "b", vec(1))
    s.put("c", "c", vec(2))
    for _ in range(10):
        s.fetch(hot)                       # hot is now MRU, cold is LRU
    s.soft_delete([cold])
    s.compact()
    s.put("d", "d", vec(3))
    s.put("e", "e", vec(4))                # forces an eviction
    assert s.fetch(hot) is not None, "compaction reset recency; the hottest entry went first"
    s.close()


def test_soft_delete_releases_its_lru_slot():
    """Without the pop, dead ids keep occupying max_size slots and the next put
    evicts a live entry that should have survived.
    """
    s = Store(":memory:", dim=DIM, max_size=3)
    ids = [s.put(f"q{i}", f"a{i}", vec(i)) for i in range(3)]
    s.fetch(ids[0])                        # the doomed id is now MRU, not LRU,
    s.soft_delete([ids[0]])                # so only a real pop frees its slot
    keep = s.put("keep", "keep", vec(9))
    assert s.fetch(ids[1]) is not None, "a live entry was evicted for a dead id"
    assert s.fetch(ids[2]) is not None
    assert s.fetch(keep) is not None
    s.close()
