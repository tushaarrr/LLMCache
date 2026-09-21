"""Shadow mode: never serve, record what would have been served.

The measuring instrument. Every later safety change is evaluated with this on
real traffic rather than on 66 hand-written pairs.
"""

import json

import pytest

from semcache import Cache, shadow_report

from .conftest import DIM, stub


def _lines(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def test_shadow_never_serves(tmp_path):
    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=log)
    c.put("q", "a")
    assert c.get("q") is None                 # identical prompt, still no serve
    rec = _lines(log)[0]
    assert rec["would_hit"] is True
    assert rec["matched_question"] == "q"
    assert rec["score"] == pytest.approx(1.0, abs=1e-5)
    c.close()


def test_shadow_records_a_miss_too(tmp_path):
    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=log)
    c.put("q", "a")
    c.get("something else entirely")
    rec = _lines(log)[-1]
    assert rec["would_hit"] is False
    assert rec["matched_question"] is None
    c.close()


def test_shadow_writes_one_line_per_query(tmp_path):
    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=log)
    c.put("q", "a")
    for _ in range(5):
        c.get("q")
    assert len(_lines(log)) == 5
    c.close()


def test_shadow_record_has_every_field(tmp_path):
    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=log)
    c.put("q", "a", session_id="acme")
    c.get("q", session_id="acme")
    rec = _lines(log)[0]
    assert set(rec) == {"prompt", "matched_question", "score", "would_hit",
                        "session_id", "ts"}
    assert rec["session_id"] == "acme"


def test_shadow_log_failure_does_not_break_get(tmp_path):
    """An unwritable log must never take down the request path it measures."""
    unwritable = tmp_path / "no_such_dir" / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=unwritable)
    c.put("q", "a")
    assert c.get("q") is None          # no exception escapes
    assert c.get("anything") is None
    c.close()


def test_shadow_off_by_default(tmp_path):
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM)
    c.put("q", "a")
    assert c.shadow is False
    assert c.get("q") == "a"           # serves normally
    c.close()


def test_shadow_applies_to_aget(tmp_path):
    import asyncio
    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=stub, dim=DIM, shadow=True,
              shadow_log=log)

    async def main():
        await c.aput("q", "a")
        return await c.aget("q")

    assert asyncio.run(main()) is None
    assert _lines(log)[0]["would_hit"] is True
    c.close()


def test_shadow_report(tmp_path):
    """Controlled vectors, not the random stub: in 8 dims two random unit
    vectors clear 0.68 often enough to make the counts nondeterministic.
    """
    import numpy as np

    vecs = {}
    for i in range(10):                       # 10 mutually orthogonal hits
        v = np.zeros(16, dtype="float32"); v[i] = 1.0
        vecs[f"q{i}"] = v
    for i in range(5):                        # 5 misses, orthogonal to all of them
        v = np.zeros(16, dtype="float32"); v[10 + i % 6] = 1.0
        vecs[f"miss{i}"] = v

    log = tmp_path / "s.jsonl"
    c = Cache(tmp_path / "c.db", embedder=vecs.__getitem__, dim=16, shadow=True,
              shadow_log=log)
    for i in range(10):
        c.put(f"q{i}", f"a{i}")
    for i in range(10):
        c.get(f"q{i}")                 # 10 would-be hits
    for i in range(5):
        c.get(f"miss{i}")              # 5 would-be misses
    c.close()

    rep = shadow_report(log)
    assert rep["total"] == 15
    assert rep["would_hit"] == 10
    assert rep["would_hit_rate"] == pytest.approx(10 / 15)
    assert sum(rep["score_distribution"].values()) == 10
    assert len(rep["top_20"]) == 10
    assert rep["top_20"][0]["score"] >= rep["top_20"][-1]["score"]
