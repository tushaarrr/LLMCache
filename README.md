# semcache

A semantic cache for LLM responses. Ask a question that *means* the same as one
you already asked, get the stored answer back instead of paying for another API
call.

```
"what is github?"          -> MISS -> call the LLM -> store
"what's GitHub"            -> HIT  (cosine 0.94) -> 0ms, $0
"how do I cook risotto"    -> MISS -> call the LLM -> store
```

Three files, ~570 lines, three dependencies. Not an LLM SDK wrapper: you call
your own model, semcache only decides whether you need to.

## Install

```bash
pip install numpy cachetools sentence-transformers
```

`numpy` and `cachetools` are required. `sentence-transformers` is the default
embedder and the only heavy dependency — swap it for any `str -> ndarray`
callable and you can drop it. sqlite3 is stdlib. No vector database, no ORM.

## Quickstart

```python
from semcache import Cache

cache = Cache("cache.db")                    # or ":memory:"

answer = cache.get("what is github?")
if answer is None:
    answer = my_llm("what is github?")       # your call, your SDK, your retries
    cache.put("what is github?", answer)

print(cache.get("what's GitHub"))            # hits the row above
```

Wrap it once and forget it:

```python
def ask(prompt):
    hit = cache.get(prompt)
    if hit is not None:
        return hit
    answer = my_llm(prompt)
    cache.put(prompt, answer)
    return answer
```

## How it works

`get` embeds the prompt, takes the top-k nearest stored vectors by cosine
similarity, drops anything below the threshold, and returns the best surviving
answer. Vectors are L2-normalized on both the write and query paths, so cosine
is a plain dot product and the search is one numpy matmul.

Embeddings are stored durably in sqlite as BLOBs; the in-memory index is a
derived cache of them. That makes restart and compaction trivial — the index is
always rebuildable from the scalar store.

## API

### `Cache(path=":memory:", embedder=None, dim=None, threshold=0.68, max_size=1000, top_k=5, skip_pattern=ID_PATTERN, max_age=None)`

| Arg | Meaning |
|---|---|
| `path` | sqlite file, or `":memory:"` |
| `embedder` | `Callable[[str], np.ndarray]`; defaults to `all-MiniLM-L6-v2` |
| `dim` | embedding dimension; inferred if omitted |
| `threshold` | cosine similarity required for a hit, `[-1, 1]` (default 0.68) |
| `max_size` | live entries before the LRU starts evicting |
| `top_k` | candidates pulled from the index per query |
| `skip_pattern` | prompts matching it never consult the cache; `None` disables |
| `max_age` | seconds before an entry goes stale; `None` means never |

- `get(prompt, threshold=None, session_id=None, max_age=None) -> str | None`
- `put(prompt, answer, session_id=None) -> int | None`
- `aget(...)` / `aput(...)` — same semantics, off the event loop
- `search(prompt, top_k=None, session_id=None, max_age=None) -> list[(score, question, answer)]` — unfiltered by threshold; the calibration tool
- `close()` — also runs on `__exit__`

Async runs the sync body in `asyncio.to_thread`. If the embedder is itself a
coroutine function it is awaited on the loop instead of blocking a worker.

## Expiry

Nothing expires unless you ask. `max_age` is in seconds, per-call or per-cache:

```python
cache = Cache("cache.db", max_age=86_400)   # sweep reclaims disk
cache.get("what is the latest python version", max_age=3600)   # hides only
```

Expiry is applied before the top-k cut, so a stale entry cannot occupy a
candidate slot. Only the instance-level `max_age` marks rows for compaction; a
per-call value hides them without reclaiming the file.

## Shadow mode

Measure before you trust. `shadow=True` never serves and records what it would
have served, one JSON line per query:

```python
cache = Cache("cache.db", shadow=True, shadow_log="shadow.jsonl")
...
from semcache import shadow_report
shadow_report("shadow.jsonl")   # hit rate, score distribution, top 20
```

## Exact repeats

An exact repeat (case- and whitespace-insensitive) resolves from a bounded dict
without touching the embedder, which is 70–85% of a hit. It still goes through
the same row lookup, so sessions, expiry and eviction all apply — the L0 layer
holds row ids, never answers.

## Calibrate the threshold on your own traffic

The default 0.68 is a starting point, not an answer. It is deliberately strict:
GPTCache's equivalent default works out to cosine **0.60**, verified by
reproducing all 61 of its hit/miss decisions in `tests/test_parity.py`. Pass
`threshold=0.60` if you want its behavior instead of a safer one. Collect real prompt pairs,
label them same-intent or not, and sweep.

**The asymmetry that should drive the choice:** a false miss costs one API call;
a false hit returns a **wrong answer to a user**, silently, and you find out from
a support ticket. Start strict and loosen.

### The limit worth knowing before you deploy

A general-purpose embedder cannot separate some pairs at any threshold:

| Family | Example | Cosine |
|---|---|---|
| direction | `convert celsius to fahrenheit` / `...fahrenheit to celsius` | **0.994** |
| negation | `is python garbage collected` / `is python not garbage collected` | **0.981** |
| unit | `set the timeout to 30 seconds` / `...30 minutes` | **0.970** |

Each is one token against a near-identical sentence. No threshold fixes this —
the remedies are a domain embedder or a cross-encoder rerank.

For opaque identifiers (ticket ids, SKUs, versions, status codes) the fix is to
not look at all, which is what `skip_pattern` does by default:

```python
ID_PATTERN = re.compile(r"\b([A-Z]{2,}-\d+|\d{3,}|v?\d+\.\d+)\b")
```

It neutralizes 7 of 31 adversarial pairs for 1 of 29 legitimate hits lost. Pass
`skip_pattern=None` to disable.

### Reranking cuts the rest roughly in half

A bi-encoder scores each sentence alone and never sees the pair. A cross-encoder
reads both together, which is what direction and negation need:

```python
from semcache import Cache, cross_encoder_reranker

cache = Cache("cache.db",
              reranker=cross_encoder_reranker(),   # quora-distilroberta
              rerank_threshold=0.0322)             # NOT on the cosine scale
```

Measured on a held-out half of the labelled set, with the threshold chosen on
the other half only:

| | recall | false hits |
|---|---|---|
| cosine ≥ 0.68 | 93% | 12/13 |
| rerank ≥ 0.0322 | 93% | **7/13** |

Same recall, roughly half the wrong answers — **and still not zero.** Three of
the survivors score 0.96+, so no threshold reaches zero at any usable recall.
Treat reranking as a large improvement, not a guarantee.

### Verify-on-hit

For the uncertain middle, hand the pair to a small model:

```python
cache = Cache("cache.db", verifier=my_llm_check, verify_band=(0.60, 0.85))
```

Above the band the score speaks for itself; below it the answer is a miss
anyway; only the band pays for a call. It fails **closed** — a verifier that
raises is a miss. `cache.band_rate()` reports what fraction actually paid.

Note the caveat that the measurement exposes: on this dataset the worst false
hits score 0.96+, i.e. *above* a default band, so catching them needs an upper
bound near 1.0 and therefore verifying nearly every hit. That is a cost
decision, not a default.

## Sessions

`session_id` isolates tenants. A prompt stored under one is invisible to
another, filtered before the top-k cut so tenants cannot starve each other:

```python
cache.put("what is our revenue", "42M", session_id="acme")
cache.get("what is our revenue", session_id="globex")   # None
cache.get("what is our revenue", session_id="acme")     # "42M"
```

## Eviction

Two tiers, because deleting from a vector index is expensive and deleting from
sqlite is not. The LRU pops the least-recently-used id and the row is marked
`state = -1`; once marked rows exceed 5000 or 10% of the store, compaction
hard-deletes them and rebuilds the index. Reads touch the LRU, so recency is
real, and it survives compaction.

## Performance

50k entries × 768 dims, excluding the embedder:

| Operation | Measured |
|---|---|
| `search` p50 | 4.2 ms |
| `compact()` (10% marked) | 82 ms |
| startup index load | 0.11 s |
| `put` amortized | 0.31 ms |

Embedding dominates below ~100k entries (10–30 ms per short prompt on CPU), so
profile the embedder before optimizing the search. Past ~200k entries, replace
the body of `Store.search` with faiss or hnswlib; the signature does not change.

## HTTP server

In-process, gunicorn with 4 workers is 4 disjoint caches and roughly 4× the miss
rate. One process owning the cache is the fix:

```bash
pip install fastapi uvicorn requests        # optional; the library never needs them
python -m semcache.server --db cache.db --port 8000
```

```python
from semcache.client import Client
cache = Client("http://localhost:8000")     # same get/put signatures as Cache
```

Routes: `POST /get`, `POST /put`, `GET /health`.

## Tests

```bash
pytest tests/ -q             # 150 unit, invariant and integration tests
pytest tests/ -q -m slow     # + performance budgets and the real embedder
python -m tests.golden_pairs # threshold calibration sweep
```

The suite is mutation-verified: 30 deliberately injected defects, 30 caught.

## Limits

- **Semantic hits are never provably safe.** Even with reranking, ~7 of 13
  adversarial holdout pairs still hit. Run `shadow=True` against your own
  traffic before trusting it in a request path.
- **Nothing expires unless you set `max_age`.** Cached facts are served forever.
- **Single writer.** One lock, one sqlite connection. The HTTP server serializes
  on writes — fine to a few hundred req/s.
- **Brute-force search.** O(n) scan, fine to ~200k entries.
- **Capacity is shared across tenants.** Isolation holds, but there is no per-tenant floor.
- **Long-context prompts are not special-cased.** Two questions over the same
  large document score ~0.99 and will collide.

## License

MIT
