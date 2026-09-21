"""HTTP front end: one process owns the cache, everyone else talks to it.

In-process, gunicorn with 4 workers means 4 disjoint caches and roughly 4x the
miss rate. That is a hit-rate bug, not a concurrency footnote, and one owner is
the fix.

fastapi and uvicorn are optional: importing semcache must not require them.

    python -m semcache.server --db cache.db --port 8000

# ponytail: one Cache behind one RLock; requests serialize on writes.
# Fine to ~hundreds of req/s. Shard by prompt hash if that ever binds.
"""

import argparse
import contextlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .cache import Cache

_cache = None


class GetRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    threshold: float | None = None
    max_age: float | None = None


class PutRequest(BaseModel):
    prompt: str
    answer: str
    session_id: str | None = None


@contextlib.asynccontextmanager
async def lifespan(app):
    yield
    if _cache is not None:
        _cache.close()


app = FastAPI(title="semcache", lifespan=lifespan)


@app.post("/get")
async def get(req: GetRequest):
    answer = await _cache.aget(req.prompt, threshold=req.threshold,
                               session_id=req.session_id, max_age=req.max_age)
    return {"hit": answer is not None, "answer": answer}


@app.post("/put")
async def put(req: PutRequest):
    row_id = await _cache.aput(req.prompt, req.answer, session_id=req.session_id)
    return {"id": row_id, "stored": row_id is not None}


@app.get("/health")
async def health():
    if _cache is None:
        raise HTTPException(status_code=503, detail="cache not initialized")
    return {"status": "ok", "entries": _cache._store.count(state=0)}


def build(**kwargs):
    """Install the one Cache the routes share. Returns it."""
    global _cache        # noqa: PLW0603 -- one process, one cache, on purpose
    _cache = Cache(**kwargs)
    return _cache


def main(argv=None):
    parser = argparse.ArgumentParser(prog="semcache.server")
    parser.add_argument("--db", default="cache.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--max-age", type=float, default=None)
    args = parser.parse_args(argv)

    import uvicorn  # noqa: PLC0415 -- optional dependency

    kwargs = {"path": args.db, "max_size": args.max_size, "max_age": args.max_age}
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    build(**kwargs)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
