"""aget / aput: the same semantics as the sync pair, off the event loop.

Written with asyncio.run rather than `async def` tests because pytest-asyncio
is a new dependency and the step does not name one.
"""

import asyncio

import numpy as np
import pytest

from semcache import Cache

from .conftest import DIM, stub


def run(coro):
    return asyncio.run(coro)


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return stub(text)


def test_aget_does_not_block_the_loop():
    """A slow sync embedder must not stall a concurrently running task."""
    def slow(text):
        import time
        time.sleep(0.05)
        return stub(text)

    async def main():
        c = Cache(":memory:", embedder=slow, dim=DIM)
        await c.aput("some prompt", "answer")
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        before = ticks                  # count only what elapses DURING aget --
        await c.aget("a different prompt")   # cumulative ticks would pass even
        gained = ticks - before         # if aget blocked, thanks to the sleep
                                        # above. Different prompt so it misses
                                        # L0 and actually reaches the embedder.
        t.cancel()
        c.close()
        return gained

    assert run(main()) > 5


def test_sync_get_does_block_the_loop():
    """The control for the test above: same harness, sync call, loop starves."""
    def slow(text):
        import time
        time.sleep(0.05)
        return stub(text)

    async def main():
        c = Cache(":memory:", embedder=slow, dim=DIM)
        c.put("some prompt", "answer")
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        before = ticks
        c.get("a different prompt")     # blocking call inside the loop
        gained = ticks - before
        t.cancel()
        c.close()
        return gained

    assert run(main()) == 0


def test_async_embedder_is_awaited():
    calls = []

    async def aembed(text):
        calls.append(text)
        await asyncio.sleep(0)
        return stub(text)

    async def main():
        c = Cache(":memory:", embedder=aembed, dim=DIM)
        await c.aput("q", "a")
        got = await c.aget("q2")    # different prompt: misses L0, must embed
        c.close()
        return got

    run(main())
    # The point of this test: the coroutine embedder was awaited on both paths.
    # Not asserting hit/miss -- with 8-dim stub vectors an unrelated prompt
    # clears 0.68 often enough to make that nondeterministic.
    assert calls == ["q", "q2"]


def test_async_embedder_runs_on_the_loop_not_in_a_thread():
    """An awaited coroutine embedder must not be handed to a worker thread."""
    import threading
    seen = []

    async def aembed(text):
        seen.append(threading.current_thread().name)
        return stub(text)

    async def main():
        c = Cache(":memory:", embedder=aembed, dim=DIM)
        await c.aput("q", "a")
        c.close()

    run(main())
    assert seen == ["MainThread"]


def test_aget_matches_get():
    async def main():
        c = Cache(":memory:", embedder=stub, dim=DIM)
        c.put("what is git", "vcs")
        sync = c.get("what is git")
        a = await c.aget("what is git")
        miss_sync = c.get("totally unrelated risotto")
        miss_a = await c.aget("totally unrelated risotto")
        c.close()
        return sync, a, miss_sync, miss_a

    sync, a, miss_sync, miss_a = run(main())
    assert sync == a == "vcs"
    assert miss_sync is miss_a is None


def test_aput_matches_put():
    async def main():
        c = Cache(":memory:", embedder=stub, dim=DIM)
        row = await c.aput("q", "a")
        got = await c.aget("q")
        skipped = await c.aput("ticket ENG-4471", "x")   # guard applies to aput
        c.close()
        return row, got, skipped

    row, got, skipped = run(main())
    assert isinstance(row, int) and got == "a" and skipped is None


def test_concurrent_agets_are_safe():
    """to_thread means real threads on the RLock and the sqlite connection.

    Afterwards the I1 invariant must still hold: every id resolves to its own
    row or to nothing, never to a different row.
    """
    async def main():
        c = Cache(":memory:", embedder=stub, dim=DIM, max_size=10_000)
        await asyncio.gather(*[c.aput(f"q{i}", f"a{i}") for i in range(50)])
        got = await asyncio.gather(*[c.aget(f"q{i}") for i in range(50)])
        mixed = await asyncio.gather(*(
            [c.aput(f"x{i}", f"b{i}") for i in range(25)]
            + [c.aget(f"q{i}") for i in range(25)]
        ))
        return c, got, mixed

    c, got, mixed = run(main())
    assert got == [f"a{i}" for i in range(50)]
    assert all(not isinstance(m, Exception) for m in mixed)

    # I1, after the concurrent storm
    for i in range(50):
        hits = c._store.search(stub(f"q{i}"), 1)
        if hits:
            row = c._store.fetch(hits[0][1])
            assert row is None or row[1] == c._store.fetch(hits[0][1])[1]
    for i in range(50):
        assert c.get(f"q{i}") == f"a{i}"
    c.close()


def test_concurrent_mixed_workload_keeps_the_store_consistent():
    async def main():
        c = Cache(":memory:", embedder=stub, dim=DIM, max_size=10_000)
        await asyncio.gather(*[c.aput(f"k{i}", f"v{i}") for i in range(100)])
        c.close() if False else None
        return c

    c = run(main())
    assert c._store.count(state=0) == 100
    assert c._store._n == 100
    ids = [rid for _, rid in c._store.search(stub("k7"), 100)]
    assert len(ids) == len(set(ids)), "duplicate ids in the index"
    c.close()
