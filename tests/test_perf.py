"""Layer 3: performance budgets.

These are BUDGETS, not measurements -- replace the numbers with what your own
hardware does. Marked slow and kept out of the default run: a perf test in the
inner loop gets deleted the first week it flakes on a loaded CI box.

    pytest tests/ -q -m slow
"""

import time

import numpy as np
import pytest

from semcache.store import Store

pytestmark = pytest.mark.slow

DIM = 768
N = 50_000


@pytest.fixture(scope="module")
def store_50k():
    rng = np.random.default_rng(0)
    s = Store(":memory:", dim=DIM, max_size=N * 2)
    vecs = rng.standard_normal((N, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i in range(N):
        s.put(f"q{i}", f"a{i}", vecs[i])
    yield s
    s.close()


def _unit(seed=0):
    q = np.random.default_rng(seed).standard_normal(DIM).astype("float32")
    return q / np.linalg.norm(q)


def test_search_budget(store_50k):
    q = _unit()
    t = time.perf_counter()
    for _ in range(100):
        store_50k.search(q, 5)
    p = (time.perf_counter() - t) / 100
    assert p < 0.040, f"search p50 {p * 1000:.1f}ms exceeds 40ms budget"


def test_fetch_budget(store_50k):
    t = time.perf_counter()
    for i in range(1000):
        store_50k.fetch(i + 1)
    p = (time.perf_counter() - t) / 1000
    assert p < 0.002, f"fetch p50 {p * 1000:.2f}ms exceeds 2ms budget"


def test_put_budget(store_50k):
    v = _unit(1)
    t = time.perf_counter()
    for i in range(200):
        store_50k.put(f"perf{i}", "a", v)
    p = (time.perf_counter() - t) / 200
    assert p < 0.050, f"put p50 {p * 1000:.1f}ms exceeds 50ms budget"


def test_put_is_not_quadratic():
    """Index append must amortize. A vstack-per-put is O(n^2) and blows the
    put budget long before 50k entries.
    """
    def elapsed(n):
        s = Store(":memory:", dim=64, max_size=n * 2)
        v = np.ones(64, dtype="float32")
        t = time.perf_counter()
        for i in range(n):
            s.put(f"q{i}", "a", v)
        out = time.perf_counter() - t
        s.close()
        return out

    # A quadratic append costs ~16x for 4x the rows; sqlite insert time dilutes
    # the ratio, so assert on per-put cost at the larger size instead, which a
    # vstack-per-put cannot meet.
    small, large = elapsed(2_000), elapsed(8_000)
    assert large / 8_000 < 2.5 * (small / 2_000), (
        f"per-put cost grew {(large / 8_000) / (small / 2_000):.1f}x when the "
        "row count grew 4x -- the index append is not amortizing"
    )


def test_startup_load_budget(tmp_path):
    path = str(tmp_path / "perf.db")
    rng = np.random.default_rng(2)
    s = Store(path, dim=DIM, max_size=N * 2)
    vecs = rng.standard_normal((N, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i in range(N):
        s.put(f"q{i}", f"a{i}", vecs[i])
    s.close()

    t = time.perf_counter()
    s = Store(path, max_size=N * 2)
    load = time.perf_counter() - t
    s.close()
    assert load < 2.0, f"startup _load {load:.2f}s exceeds 2s budget"


def test_compact_budget(tmp_path):
    rng = np.random.default_rng(3)
    s = Store(":memory:", dim=DIM, max_size=N * 2)
    vecs = rng.standard_normal((N, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [s.put(f"q{i}", f"a{i}", vecs[i]) for i in range(N)]
    s.soft_delete(ids[: N // 10])

    t = time.perf_counter()
    s.compact()
    took = time.perf_counter() - t
    s.close()
    assert took < 0.500, f"compact {took * 1000:.0f}ms exceeds 500ms budget"
