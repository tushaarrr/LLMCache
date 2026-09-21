"""Layer 1 invariants that live in the store: the id contract, deletes,
compaction, restart. A violation here is a bug, always.
"""

import numpy as np
import pytest

from semcache.store import Store

from .conftest import DIM, vec


# -- I1 ---------------------------------------------------------------------
def test_index_never_points_at_wrong_row(store):
    """Right row or no row -- never a different row.

    The highest-value test in the suite: the failure it catches is silently
    serving someone else's cached answer.
    """
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(100)]
    for i, row_id in enumerate(ids):
        row = store.fetch(row_id)
        assert row is None or row[0] == f"q{i}"


def test_search_resolves_to_the_planted_row(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(1000)]
    for i in (0, 7, 500, 999):
        score, row_id = store.search(vec(i), 5)[0]
        assert row_id == ids[i]
        assert store.fetch(row_id)[0] == f"q{i}"
        assert score == pytest.approx(1.0, abs=1e-5)


def test_search_is_ordered_best_first(store):
    for i in range(50):
        store.put(f"q{i}", f"a{i}", vec(i))
    scores = [s for s, _ in store.search(vec(3), 10)]
    assert scores == sorted(scores, reverse=True)


def test_ties_resolve_to_the_lower_id(store):
    """Documented API behavior: identical prompts stored twice, lower id wins."""
    first = store.put("q", "a1", vec(0))
    second = store.put("q", "a2", vec(0))
    assert first < second
    assert store.search(vec(0), 2)[0][1] == first


# -- I2 ---------------------------------------------------------------------
def test_compaction_preserves_survivors(store):
    """Both directions: survivors still fetch AND still appear in search.

    The second assertion is the one people forget -- a compaction that keeps
    sqlite correct but rebuilds the index off stale ids passes only the first.
    """
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(100)]
    store.soft_delete(ids[:50])
    store.compact()
    for i, row_id in enumerate(ids[50:], start=50):
        assert store.fetch(row_id) is not None
        assert row_id in [rid for _, rid in store.search(vec(i), 5)]


def test_compaction_hard_deletes_marked_rows(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(20)]
    store.soft_delete(ids[:10])
    assert store.count(all=True) == 20
    store.compact()
    assert store.count(all=True) == 10
    assert store.count(state=0) == 10


def test_compaction_is_idempotent(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(20)]
    store.soft_delete(ids[:5])
    store.compact()
    store.compact()
    assert store.count(state=0) == 15
    assert store.fetch(ids[19]) is not None


# -- I3 ---------------------------------------------------------------------
def test_soft_deleted_invisible(store):
    row_id = store.put("q", "a", vec(0))
    store.soft_delete([row_id])
    assert store.fetch(row_id) is None
    assert store.count(state=0) == 0


def test_soft_deleted_may_still_be_in_the_raw_index(store):
    """Expected, and exactly why a None fetch is a normal control path."""
    row_id = store.put("q", "a", vec(0))
    store.soft_delete([row_id])
    assert [rid for _, rid in store.search(vec(0), 5)] == [row_id]
    assert store.fetch(row_id) is None


def test_soft_delete_half_none_come_back(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(200)]
    dead = set(ids[::2])
    store.soft_delete(sorted(dead))
    for row_id in ids:
        assert (store.fetch(row_id) is None) is (row_id in dead)


# -- I6 ---------------------------------------------------------------------
def test_sqlite_is_written_before_the_index(store, monkeypatch):
    """Pins the write ORDER, not just that a crash is survivable. If put wrote
    the index first, a failing _insert would leave a vector id with no row --
    the dangling entry the order exists to prevent.
    """
    def boom(*_a, **_k):
        raise RuntimeError("crash")

    before = store._n
    monkeypatch.setattr(store, "_insert", boom)
    with pytest.raises(RuntimeError):
        store.put("q", "a", vec(7))
    monkeypatch.undo()
    assert store._n == before, "index grew despite the sqlite write failing"


def test_crash_between_writes(store, monkeypatch):
    """sqlite -> index -> LRU is the write order precisely so a crash leaves an
    unreachable orphan row rather than a dangling index entry.
    """
    def boom(*_a, **_k):
        raise RuntimeError("crash")

    good = store.put("q1", "a1", vec(1))
    monkeypatch.setattr(store, "_index_append", boom)
    with pytest.raises(RuntimeError):
        store.put("q2", "a2", vec(2))
    monkeypatch.undo()

    assert store.fetch(good) is not None
    store.compact()
    assert store.search(vec(2), 5) == [] or all(
        store.fetch(rid) is not None for _, rid in store.search(vec(2), 5)
    )


def test_orphan_row_is_recovered_not_lost(store, monkeypatch):
    """The embedding is durable in sqlite, so the next index rebuild heals the
    orphan instead of discarding a perfectly good entry.
    """
    def boom(*_a, **_k):
        raise RuntimeError("crash")

    monkeypatch.setattr(store, "_index_append", boom)
    with pytest.raises(RuntimeError):
        store.put("orphan", "answer", vec(42))
    monkeypatch.undo()

    assert store.search(vec(42), 5) == []      # not in the index yet
    store.compact()
    hits = store.search(vec(42), 5)
    assert hits and store.fetch(hits[0][1])[0] == "orphan"


# -- I8 ---------------------------------------------------------------------
def test_unnormalized_input_still_scores_correctly(store):
    """If normalization is missing on either path this returns 25.0 and every
    threshold in the system is meaningless.
    """
    v = np.array([3.0, 4.0] + [0.0] * (DIM - 2), dtype="float32")   # norm 5
    store.put("q", "a", v)
    score, _ = store.search(v, 1)[0]
    assert abs(score - 1.0) < 1e-5


def test_normalization_applied_on_write_path_only_input(store):
    """Query normalized, stored vector not: still 1.0 only if writes normalize."""
    big = np.array([10.0] + [0.0] * (DIM - 1), dtype="float32")
    unit = np.array([1.0] + [0.0] * (DIM - 1), dtype="float32")
    store.put("q", "a", big)
    assert abs(store.search(unit, 1)[0][0] - 1.0) < 1e-5


def test_zero_vector_does_not_produce_nan(store):
    store.put("q", "a", np.zeros(DIM, dtype="float32"))
    score, _ = store.search(vec(0), 1)[0]
    assert not np.isnan(score)


# -- restart / persistence --------------------------------------------------
def test_reopen_rebuilds_index(tmp_path):
    path = str(tmp_path / "c.db")
    s = Store(path, dim=DIM)
    ids = [s.put(f"q{i}", f"a{i}", vec(i)) for i in range(50)]
    s.close()

    s = Store(path, dim=DIM)
    assert s.count(state=0) == 50
    assert s.search(vec(7), 1)[0][1] == ids[7]
    assert s.fetch(ids[7])[1] == "a7"
    s.close()


def test_reopen_infers_dim_from_stored_blobs(tmp_path):
    path = str(tmp_path / "c.db")
    s = Store(path, dim=DIM)
    row_id = s.put("q", "a", vec(0))
    s.close()

    s = Store(path)                       # no dim given
    assert s.search(vec(0), 1)[0][1] == row_id
    s.close()


def test_reopen_drops_soft_deleted_from_the_index(tmp_path):
    path = str(tmp_path / "c.db")
    s = Store(path, dim=DIM)
    ids = [s.put(f"q{i}", f"a{i}", vec(i)) for i in range(10)]
    s.soft_delete(ids[:5])
    s.close()

    s = Store(path)
    hits = s.search(vec(0), 10)
    assert all(rid in ids[5:] for _, rid in hits)          # dead ones are gone
    # ranking survives the rebuild: same order a brute-force scan would give
    expected = [rid for rid, _ in sorted(
        ((ids[i], float(np.dot(vec(i), vec(0)))) for i in range(5, 10)),
        key=lambda t: -t[1])]
    assert [rid for _, rid in hits] == expected
    s.close()


# -- misc contract ----------------------------------------------------------
def test_empty_store_search_returns_empty(store):
    assert store.search(vec(0), 5) == []


def test_fetch_unknown_id_returns_none(store):
    assert store.fetch(12345) is None


def test_top_k_is_capped_by_population(store):
    for i in range(3):
        store.put(f"q{i}", f"a{i}", vec(i))
    assert len(store.search(vec(0), 100)) == 3


def test_dim_mismatch_is_rejected_before_the_insert(store):
    """numpy raises ValueError on its own further down, so asserting only on the
    exception type passes with the guard deleted -- and by then the bad row is
    already committed and the store can never be reopened.
    """
    store.put("ok", "a", vec(0))
    with pytest.raises(ValueError):
        store.put("q", "a", np.ones(DIM + 1, dtype="float32"))
    assert store.count(all=True) == 1      # nothing was written


def test_search_dim_mismatch_returns_empty(store):
    store.put("q", "a", vec(0))
    assert store.search(np.ones(DIM + 1, dtype="float32"), 5) == []


@pytest.mark.parametrize("n_ties,k", [(10, 5), (20, 5), (100, 5), (500, 5)])
def test_ties_resolve_to_the_lower_id_beyond_top_k(n_ties, k):
    """The API promises "ties resolve to the lower id", and `put` deliberately
    does not dedupe, so exact ties are an expected state.

    argpartition keeps an ARBITRARY k of the tied rows, so sorting only that
    subset returns whichever id it happened to keep -- id 7 of 10 at k=5, not
    id 1. The tie has to be widened to every row at the cut score first.
    """
    s = Store(":memory:", dim=DIM, max_size=10_000)
    ids = [s.put(f"dup{i}", f"a{i}", vec(0)) for i in range(n_ties)]
    hits = s.search(vec(0), k)
    assert len(hits) == k
    assert hits[0][1] == min(ids)
    assert [rid for _, rid in hits] == sorted(ids)[:k]
    s.close()


def test_max_size_must_be_positive():
    with pytest.raises(ValueError):
        Store(":memory:", dim=DIM, max_size=0)


# -- rowid reuse (I1) --------------------------------------------------------
def test_compaction_never_reuses_an_id(store):
    """A plain INTEGER PRIMARY KEY is a rowid alias and sqlite hands the id of a
    hard-deleted row to the next insert. An id captured before a compaction
    would then fetch a live, valid, COMPLETELY UNRELATED row -- the None guard
    never fires and a wrong answer is served. This is I1, which is binary.
    """
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(5)]
    doomed = ids[-1]                                  # the HIGHEST rowid
    store.soft_delete([doomed])
    store.compact()
    fresh = store.put("unrelated question", "unrelated answer", vec(99))
    assert fresh != doomed
    assert store.fetch(doomed) is None                # stale id resolves to nothing


def test_ids_are_monotonic_across_many_compactions(store):
    seen = set()
    for round_ in range(10):
        ids = [store.put(f"r{round_}q{i}", "a", vec(i)) for i in range(10)]
        assert not (seen & set(ids)), "id reuse across compaction"
        seen |= set(ids)
        store.soft_delete(ids[5:])
        store.compact()


# -- normalization edge cases -------------------------------------------------
def test_non_finite_vector_degrades_to_a_miss(store):
    """inf/NaN cannot be normalized. Returning zeros makes it score 0.0 against
    everything -- a guaranteed miss, rather than a NaN that comparisons accept.
    """
    store.put("q", "a", vec(0))
    for bad in (np.full(DIM, np.inf, dtype="float32"),
                np.full(DIM, np.nan, dtype="float32")):
        scores = [s for s, _ in store.search(bad, 5)]
        assert all(np.isfinite(s) for s in scores)
        assert all(s == 0.0 for s in scores)


def test_huge_vector_does_not_overflow_to_zero(store):
    """A float32 norm overflows to inf past ~1.8e19, and v / inf is all zeros."""
    big = np.zeros(DIM, dtype="float32")
    big[0] = 3e19
    store.put("q", "a", big)
    assert store.search(np.eye(DIM, dtype="float32")[0], 1)[0][0] == pytest.approx(1.0)


def test_count_all_includes_soft_deleted(store):
    ids = [store.put(f"q{i}", f"a{i}", vec(i)) for i in range(10)]
    store.soft_delete(ids[:4])
    assert store.count(state=0) == 6
    assert store.count(state=-1) == 4
    assert store.count(all=True) == 10
