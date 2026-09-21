"""The get/put contract: hit, miss, threshold, sessions, restart.

Ported from the original's test_core.py / adapter/test_api.py / test_session.py,
minus everything that exercised a vendor SDK wrapper semcache does not have.
"""

import numpy as np
import pytest

from semcache import Cache

from .conftest import DIM, at_cosine, stub, vec


# -- basic flow (test_core.py) ----------------------------------------------
def test_miss_on_empty_cache(cache):
    assert cache.get("anything") is None


def test_put_then_get_exact(cache):
    cache.put("what is github?", "a code host")
    assert cache.get("what is github?") == "a code host"


def test_unrelated_prompt_misses(cache):
    cache.put("what is github?", "a code host")
    assert cache.get("how do I cook risotto") is None


def test_put_returns_row_id(cache):
    assert isinstance(cache.put("q", "a"), int)


def test_put_does_not_deduplicate(cache):
    """Documented: no dedup on write. Two rows, get resolves to the lower id."""
    first = cache.put("q", "a1")
    second = cache.put("q", "a2")
    assert first != second
    assert cache.get("q") == "a1"


# -- threshold (test_api.py) -------------------------------------------------
def test_threshold_boundary_flips_at_the_configured_value():
    """Plant two vectors at a known cosine; the hit must flip across it."""
    a, b = at_cosine(0.80)
    c = Cache(":memory:", embedder={"a": a, "b": b}.__getitem__, dim=DIM,
              threshold=0.79)
    c.put("a", "answer")
    assert c.get("b") == "answer"          # 0.80 >= 0.79
    c.threshold = 0.81
    assert c.get("b") is None              # 0.80 <  0.81
    c.close()


def test_per_call_threshold_overrides_instance(cache):
    cache.put("q", "a")
    assert cache.get("q", threshold=1.01) is None
    assert cache.get("q", threshold=0.0) == "a"
    assert cache.get("q") == "a"           # instance default untouched


def test_default_threshold_is_the_gptcache_translation():
    """faiss L2 <= 0.8 over normalized vectors == cosine >= 0.68."""
    from semcache.cache import DEFAULT_THRESHOLD

    assert DEFAULT_THRESHOLD == pytest.approx(1 - 0.8 ** 2 / 2)
    assert Cache(":memory:", embedder=stub, dim=DIM).threshold == 0.68


def test_below_threshold_neighbour_is_a_miss():
    a, b = at_cosine(0.50)
    c = Cache(":memory:", embedder={"a": a, "b": b}.__getitem__, dim=DIM)
    c.put("a", "answer")
    assert c.get("b") is None
    c.close()


# -- I4: a stale hit degrades to a miss --------------------------------------
def test_stale_hit_is_a_miss(cache):
    """Eviction racing a read must degrade to a miss, never raise into the
    caller's request path.
    """
    cache.put("hello world", "hi")
    row_id = cache._store.search(cache._embed("hello world"), 1)[0][1]
    cache._store.soft_delete([row_id])
    assert cache.get("hello world") is None


def test_stale_hit_falls_through_to_the_next_candidate(cache):
    """The dead top candidate must not shadow a live lower-scoring one."""
    a, b = at_cosine(0.95)
    c = Cache(":memory:", embedder={"a": a, "b": b, "q": a}.__getitem__, dim=DIM)
    dead = c.put("a", "dead answer")
    c.put("b", "live answer")
    c._store.soft_delete([dead])
    assert c.get("q") == "live answer"
    c.close()


# -- I5: restart is transparent ----------------------------------------------
def test_restart(tmp_path):
    path = tmp_path / "c.db"
    with Cache(path, embedder=stub, dim=DIM) as c:
        c.put("what is git", "a version control system")
    with Cache(path, embedder=stub, dim=DIM) as c:
        assert c.get("what is git") == "a version control system"


def test_restart_without_declaring_dim(tmp_path):
    path = tmp_path / "c.db"
    with Cache(path, embedder=stub) as c:
        c.put("what is git", "a version control system")
    with Cache(path, embedder=stub) as c:
        assert c.get("what is git") == "a version control system"


# -- F10: tenant isolation ----------------------------------------------------
def test_no_cross_tenant_hits(cache):
    """A leak here is a data breach, not a cache miss."""
    cache.put("what is our revenue", "42M", session_id="acme")
    assert cache.get("what is our revenue", session_id="globex") is None
    assert cache.get("what is our revenue", session_id="acme") == "42M"


def test_sessionless_entry_is_not_served_to_a_session(cache):
    cache.put("what is our revenue", "42M")
    assert cache.get("what is our revenue", session_id="acme") is None
    assert cache.get("what is our revenue") == "42M"


def test_session_entry_is_not_served_sessionlessly(cache):
    cache.put("what is our revenue", "42M", session_id="acme")
    assert cache.get("what is our revenue") is None


def test_sessions_do_not_shadow_each_other(cache):
    cache.put("q", "acme answer", session_id="acme")
    cache.put("q", "globex answer", session_id="globex")
    assert cache.get("q", session_id="acme") == "acme answer"
    assert cache.get("q", session_id="globex") == "globex answer"


# -- search(), the calibration tool ------------------------------------------
def test_search_is_unfiltered_by_threshold(cache):
    cache.put("q", "a")
    hits = cache.search("totally different", top_k=5)
    assert hits and hits[0][0] < cache.threshold
    assert hits[0][1] == "q" and hits[0][2] == "a"


def test_search_is_ordered_and_shaped(cache):
    for i in range(5):
        cache.put(f"q{i}", f"a{i}")
    hits = cache.search("q2")
    assert [h[1] for h in hits][0] == "q2"
    assert [h[0] for h in hits] == sorted([h[0] for h in hits], reverse=True)
    assert all(len(h) == 3 for h in hits)


def test_search_skips_dead_rows(cache):
    row_id = cache.put("q", "a")
    cache._store.soft_delete([row_id])
    assert cache.search("q") == []


# -- lifecycle ----------------------------------------------------------------
def test_context_manager_closes(tmp_path):
    with Cache(tmp_path / "c.db", embedder=stub, dim=DIM) as c:
        c.put("q", "a")
    with pytest.raises(Exception):
        c._store.count(state=0)


def test_embedder_is_a_plain_callable():
    """The whole embedding-backend axis is one argument."""
    calls = []

    def embed(text):
        calls.append(text)
        return stub(text)

    c = Cache(":memory:", embedder=embed, dim=DIM)
    c.put("q", "a")
    assert c.get("q") == "a"
    assert calls == ["q", "q"]
    c.close()


def test_default_embedder_is_not_loaded_on_import():
    """Importing semcache must not pull in sentence-transformers.

    Has to run in a subprocess: monkeypatching sys.modules in-process cannot see
    the import move to module scope, because semcache.cache is already imported.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, semcache; "
         "print('sentence_transformers' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", out.stdout


def test_list_input_embedder_is_accepted():
    """An embedder returning a plain list must work, not just an ndarray."""
    c = Cache(":memory:", embedder=lambda t: list(stub(t)), dim=DIM)
    c.put("q", "a")
    assert c.get("q") == "a"
    c.close()


# -- the identifier guard -----------------------------------------------------
def test_opaque_identifier_prompts_never_consult_the_cache(cache):
    """No threshold separates 'ticket ENG-4471' from 'ENG-4472' (0.99), so the
    only correct move is not to look.
    """
    cache.put("status of ticket ENG-4471", "closed")
    assert cache.get("status of ticket ENG-4471") is None
    assert cache.get("status of ticket ENG-4472") is None


def test_identifier_guard_families(cache):
    for prompt in ("what does error code 401 mean", "what is CVE-2021-44228",
                   "look up SKU AX-9920", "upgrade to v2.14", "convert 1000 USD"):
        cache.put(prompt, "answer")
        assert cache.get(prompt) is None, prompt


def test_identifier_guard_does_not_overmatch(cache):
    """Ordinary prose with small numbers must still cache."""
    for prompt in ("what is github?", "python 3 tips", "explain HTTP/2",
                   "how do I center a div"):
        cache.put(prompt, f"answer for {prompt}")
        assert cache.get(prompt) == f"answer for {prompt}", prompt


def test_identifier_guard_can_be_disabled():
    c = Cache(":memory:", embedder=stub, dim=DIM, skip_pattern=None)
    c.put("status of ticket ENG-4471", "closed")
    assert c.get("status of ticket ENG-4471") == "closed"
    c.close()


def test_identifier_guard_refuses_the_write_too(cache):
    """Guarding only the read is worse than not guarding at all: get always
    misses on an id-bearing prompt, so the caller's get-miss-put loop stores a
    fresh row every time, and those rows are candidates for queries that carry
    no identifier of their own.
    """
    assert cache.put("status of ticket ENG-4471", "ticket 4471 is closed") is None
    assert cache._store.count(all=True) == 0


def test_identifier_bearing_rows_are_not_served_to_plain_queries(cache):
    """The wrong-answer path the write-side guard closes."""
    cache.put("status of ticket ENG-4471", "ticket 4471 is closed")
    assert cache.get("status of ticket") is None


def test_threshold_boundary_is_inclusive():
    """The rule is `score >= threshold`. Probing 0.79/0.81
    around a 0.80 pair cannot tell >= from >; equality has to be hit exactly.
    0.5 is exactly representable in float32, so the dot product lands on it.
    """
    a, b = at_cosine(0.5)
    c = Cache(":memory:", embedder={"a": a, "b": b}.__getitem__, dim=DIM,
              threshold=0.5, skip_pattern=None)
    c.put("a", "answer")
    assert c._store.search(c._embed("b"), 1)[0][0] == 0.5     # exactly on it
    assert c.get("b") == "answer"                             # >= must hit
    c.threshold = np.nextafter(np.float32(0.5), np.float32(1.0))
    assert c.get("b") is None
    c.close()


def test_non_finite_embedding_degrades_to_a_miss(cache):
    """_normalize turns un-normalizable input into zeros, so it scores 0.0."""
    bad = {"stored": np.array([1.0] + [0.0] * (DIM - 1), dtype="float32"),
           "query": np.array([np.inf] + [0.0] * (DIM - 1), dtype="float32")}
    c = Cache(":memory:", embedder=bad.__getitem__, dim=DIM, skip_pattern=None)
    c.put("stored", "answer")
    assert c.get("query") is None
    c.close()


def test_nan_score_is_rejected_by_the_accept_test(cache, monkeypatch):
    """Defense in depth for the comparison itself: `nan < limit` is False, so an
    accept test written as `not (score < limit)` would serve a NaN as a hit.
    Injected at the search boundary because _normalize now makes NaN scores
    unreachable through the public API.
    """
    row_id = cache.put("q", "a")
    monkeypatch.setattr(cache._store, "search",
                        lambda *_a, **_k: [(float("nan"), row_id)])
    assert cache.get("q") is None


def test_tenant_is_not_starved_out_of_its_own_entry():
    """Filtering sessions AFTER the top-k cut lets one busy tenant consume the
    whole candidate budget, so a quiet tenant misses an entry it stored itself.
    """
    a, _ = at_cosine(0.99)
    c = Cache(":memory:", embedder=lambda t: a, dim=DIM, top_k=5)
    for i in range(50):
        c.put(f"noise{i}", f"n{i}", session_id="loud")
    c.put("mine", "my answer", session_id="quiet")
    assert c.get("mine", session_id="quiet") == "my answer"
    c.close()


def test_session_ids_are_stringified_consistently(cache):
    """sqlite TEXT affinity would coerce them anyway; doing it at the boundary
    means the stored value and the query value can never disagree.
    """
    cache.put("q", "a", session_id=42)
    assert cache.get("q", session_id="42") == "a"
    assert cache.get("q", session_id=42) == "a"
    assert cache.get("q", session_id="43") is None


def test_search_sees_session_entries():
    """search() is the calibration tool; a session-blind index would make it
    return [] for everything stored under a tenant.
    """
    c = Cache(":memory:", embedder=stub, dim=DIM)
    c.put("q", "a", session_id="acme")
    assert c.search("q", session_id="acme")[0][2] == "a"
    assert c.search("q") == []
    c.close()
