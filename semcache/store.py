"""sqlite + vector index + LRU, all under one lock.

The three structures share one id space and one owner. The original splits them
into CacheStorage / VectorBase / EvictionBase and then has to keep three
independently-mutable things agreeing about one id space; every hard bug in that
design lives in the seams. One owner, one lock, no seams.

sqlite is the durable copy and the index is a derived cache of it -- which is
what makes compaction and restart trivial instead of a migration problem.
"""

import sqlite3
import threading
import time

import numpy as np

from .evict import MAX_MARK_COUNT, MAX_MARK_RATE, LRU, should_compact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    -- AUTOINCREMENT is load-bearing, not decoration. A plain INTEGER PRIMARY KEY
    -- is a rowid alias, and sqlite REUSES the rowid of a hard-deleted row. Since
    -- compact() hard-deletes, an id captured before a compaction could come back
    -- bound to an unrelated entry -- and fetch would return it, alive and valid,
    -- so the `row is None` guard never fires and a wrong answer is served.
    -- That is invariant I1 ("never resolves to a DIFFERENT row"), which is binary.
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT    NOT NULL,
    answer     TEXT    NOT NULL,
    embedding  BLOB    NOT NULL,
    session_id TEXT,
    state      INTEGER NOT NULL DEFAULT 0,   -- 0 live, -1 soft-deleted
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state ON entries(state);
"""

_MIN_CAPACITY = 1024


def _session_key(session_id):
    """sqlite's TEXT affinity would coerce 42 and "42" into the same tenant, which
    is a cross-tenant hit, not a miss. Stringify once, at the boundary, so the
    stored value and the query value can never disagree.
    """
    return None if session_id is None else str(session_id)


def _normalize(emb):
    """L2-normalize. Done here, in one place, so the write and query paths can
    never disagree -- once normalized, cosine similarity is a plain dot product.
    """
    v = np.asarray(emb, dtype="float32").ravel()
    # float64 accumulation: a float32 norm overflows to inf once any component
    # exceeds ~1.8e19, and v / inf would silently become an all-zero vector.
    n = float(np.sqrt(np.dot(v.astype(np.float64), v.astype(np.float64))))
    if not np.isfinite(n) or n == 0.0:
        # Un-normalizable (zero, inf or NaN input). Zeros score 0.0 against
        # everything, so this degrades to a guaranteed miss rather than a NaN
        # score that comparison operators would silently accept as a hit.
        return np.zeros_like(v)
    return (v / n).astype("float32")


class Store:
    """Owns the data. `put` / `search` / `fetch` / `soft_delete` / `compact`.

    :param path: sqlite file, or ":memory:"
    :param dim: embedding dimension; inferred from stored rows or the first put
    :param max_size: live entries before the LRU starts evicting
    :param max_mark_count, max_mark_rate: when maybe_compact() decides to fire
    """

    def __init__(self, path=":memory:", dim=None, max_size=1000,
                 max_mark_count=MAX_MARK_COUNT, max_mark_rate=MAX_MARK_RATE,
                 max_age=None, now=time.time):
        # Re-entrant: the LRU's eviction hook calls back into soft_delete while
        # put/fetch already hold the lock. A plain Lock deadlocks there.
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._dim = dim
        self._max_size = max_size
        self._max_mark_count = max_mark_count
        self._max_mark_rate = max_mark_rate
        self._max_age = max_age
        self._now = now
        self._n = 0
        self._ids = np.zeros(0, dtype=np.int64)
        self._vecs = np.zeros((0, dim or 0), dtype=np.float32)
        self._sess = np.full(0, None, dtype=object)
        self._created = np.zeros(0, dtype=np.float64)
        self._has_sessions = False
        self._lru = None
        self._load()

    # -- index, derived from sqlite ------------------------------------------
    def _load(self):
        """Rebuild the in-memory index and the LRU from sqlite.

        Called on open and after every compaction. Because the embeddings are
        stored durably as BLOBs, this is the only rebuild path the store needs
        -- a design that keeps vectors only in the index needs a second rebuild
        path, because that state is not regenerable from the scalar store.

        The LRU is REUSED, not rebuilt, when one already exists. Rebuilding it
        from "ORDER BY id" would replace recency with insertion order, and since
        compact() runs from put(), the LRU would collapse to FIFO every few
        thousand writes and start evicting the hottest entries first (F3, "hits
        vanish after a while"). It needs no rebuild anyway: compaction only
        hard-deletes rows that soft_delete already popped out of it.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT id, embedding, session_id, created_at FROM entries"
                " WHERE state = 0 ORDER BY id"
            ).fetchall()
            if rows and self._dim is None:
                self._dim = len(rows[0][1]) // np.dtype("float32").itemsize

            dim = self._dim or 0
            self._n = len(rows)
            cap = max(self._n, _MIN_CAPACITY)
            self._ids = np.zeros(cap, dtype=np.int64)
            self._vecs = np.zeros((cap, dim), dtype=np.float32)
            self._sess = np.full(cap, None, dtype=object)
            self._created = np.zeros(cap, dtype=np.float64)
            self._has_sessions = False
            if rows:
                self._ids[: self._n] = [r[0] for r in rows]
                self._vecs[: self._n] = np.frombuffer(
                    b"".join(r[1] for r in rows), dtype=np.float32
                ).reshape(self._n, dim)
                self._sess[: self._n] = [r[2] for r in rows]
                self._created[: self._n] = [r[3] for r in rows]
                self._has_sessions = any(r[2] is not None for r in rows)

            if self._lru is None:
                # Startup: no recency exists yet, so id order is as good as any.
                # A fresh instance, never .clear() -- see LRU's docstring.
                # Re-adding more live ids than max_size re-bounds the live set
                # through on_evict, which is correct after a reopen.
                self._lru = LRU(self._max_size, self.soft_delete)
                known = set()
            else:
                known = set(self._lru)          # post-compaction: keep recency
            for row_id in self._ids[: self._n].tolist():
                if row_id not in known:
                    self._lru[row_id] = True    # orphan of a crashed put

    def _index_append(self, row_id, emb, session_id=None, created_at=0.0):
        """Append to the index, growing by doubling.

        Amortized O(1). A vstack per put would be O(n^2) and would blow the put
        budget long before the scan does.
        """
        if self._n == len(self._ids) or self._vecs.shape[1] != self._dim:
            cap = max(_MIN_CAPACITY, self._n * 2)
            ids = np.zeros(cap, dtype=np.int64)
            vecs = np.zeros((cap, self._dim), dtype=np.float32)
            sess = np.full(cap, None, dtype=object)
            created = np.zeros(cap, dtype=np.float64)
            ids[: self._n] = self._ids[: self._n]
            sess[: self._n] = self._sess[: self._n]
            created[: self._n] = self._created[: self._n]
            if self._vecs.shape[1] == self._dim:
                vecs[: self._n] = self._vecs[: self._n]
            self._ids, self._vecs, self._sess = ids, vecs, sess
            self._created = created
        self._ids[self._n] = row_id
        self._vecs[self._n] = emb
        self._sess[self._n] = session_id
        self._created[self._n] = created_at
        self._n += 1

    # -- writes ---------------------------------------------------------------
    def _insert(self, question, answer, emb, session_id, created_at):
        cur = self._db.execute(
            "INSERT INTO entries (question, answer, embedding, session_id, state,"
            " created_at) VALUES (?, ?, ?, ?, 0, ?)",
            (question, answer, emb.tobytes(), session_id, created_at),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def put(self, question, answer, emb, session_id=None):
        """Insert one entry. Returns its id.

        The write order is sqlite -> index -> LRU and it must be this order. The
        three writes are not atomic, and a crash can land between any two:

          after (1): a row with no vector -- unreachable by search, harmless,
                     and healed by the next index rebuild
          after (2): consistent; the LRU rebuilds from sqlite on startup

        Reverse it and a crash leaves a vector id with no row, which search will
        happily return and every get then has to defend against.
        """
        with self._lock:
            emb = _normalize(emb)
            session_id = _session_key(session_id)
            if self._dim is None:
                self._dim = int(emb.shape[0])
            elif emb.shape[0] != self._dim:
                raise ValueError(
                    f"embedding dim {emb.shape[0]} != store dim {self._dim}"
                )
            created_at = self._now()
            row_id = self._insert(question, answer, emb, session_id,
                                  created_at)                          # 1. sqlite
            self._index_append(row_id, emb, session_id, created_at)    # 2. vector
            self._lru[row_id] = True                                   # 3. LRU
            self._has_sessions = self._has_sessions or session_id is not None
            return row_id

    def soft_delete(self, ids):
        """Tier 1. Mark rows dead; the vectors stay in the index and search may
        still return them -- fetch is what filters them out.
        """
        ids = [(int(i),) for i in ids]
        if not ids:
            return
        with self._lock:
            self._db.executemany("UPDATE entries SET state = -1 WHERE id = ?", ids)
            self._db.commit()
            for (row_id,) in ids:
                # Keeps max_size honest when soft_delete is called directly.
                # During eviction the key is already gone, so this is a no-op.
                self._lru.pop(row_id, None)

    # -- reads -----------------------------------------------------------------
    def _cutoff(self, max_age):
        """Oldest created_at still considered fresh. -inf means never expire."""
        if max_age is None:
            max_age = self._max_age
        return -np.inf if max_age is None else self._now() - max_age

    def search(self, emb, top_k=5, session_id=None, max_age=None):
        """Top-k nearest live-or-dead ids, best first. (score, id) pairs.

        Returns index entries, not rows: a returned id may resolve to nothing.
        Callers must treat that as a miss, not an error.

        The session filter is applied BEFORE the top-k cut, not after. Filtering
        afterwards leaks no data, but it lets one busy tenant consume the whole
        candidate budget so that every other tenant misses on entries it stored
        itself -- and each miss writes another duplicate, so the starvation
        compounds. Costs nothing until something is actually stored under a
        session id.
        """
        with self._lock:
            if self._n == 0 or self._dim is None:
                return []
            q = _normalize(emb)
            if q.shape[0] != self._dim:
                return []
            k = min(top_k, self._n)
            if k <= 0:
                return []
            scores = self._vecs[: self._n] @ q          # cosine; both normalized
            ids = self._ids[: self._n]
            if self._has_sessions:
                mine = self._sess[: self._n] == _session_key(session_id)
                if not mine.any():
                    return []
                scores = np.where(mine, scores, -np.inf)
            # Expiry is masked BEFORE the top-k cut for the same reason sessions
            # are: an expired row that outscores live ones would otherwise eat a
            # candidate slot and hide a perfectly good entry behind it.
            cutoff = self._cutoff(max_age)
            if np.isfinite(cutoff):
                fresh = self._created[: self._n] >= cutoff
                if not fresh.any():
                    return []
                scores = np.where(fresh, scores, -np.inf)
            # argpartition, not argsort: you want the top 5 of 50k, not 50k sorted.
            top = np.argpartition(-scores, k - 1)[:k]
            # argpartition takes an ARBITRARY k of the rows tied at the cut, so
            # widening to every row at that score is what makes the documented
            # "ties resolve to the lower id" true when the tie is bigger than k.
            # Costs one extra O(n) pass and only sorts the tied group.
            cut = scores[top].min()
            tied = np.flatnonzero(scores == cut)
            if tied.size > (scores[top] == cut).sum():
                top = np.union1d(top[scores[top] > cut], tied)
            # lexsort's last key is primary: best score first, lower id on ties.
            top = top[np.lexsort((ids[top], -scores[top]))][:k]
            return [(float(scores[i]), int(ids[i])) for i in top
                    if np.isfinite(scores[i])]

    def fetch(self, row_id, session_id=None, max_age=None):
        """The row behind an id, or None if it is gone.

        None is a normal control path -- the entry may have been evicted,
        compacted away, or orphaned by a crash. Never raise for it.

        session_id is matched exactly, NULL included ("IS", not "="). A prompt
        stored under one tenant is invisible to another; in a multi-tenant
        deployment the alternative is a data leak, not a cache miss.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT question, answer, session_id FROM entries"
                " WHERE id = ? AND state = 0 AND session_id IS ?"
                " AND created_at >= ?",
                (int(row_id), _session_key(session_id), self._cutoff(max_age)),
            ).fetchone()
            if row is not None:
                self._lru[int(row_id)] = True     # recency is only real if reads touch it
            return row

    def count(self, state=0, all=False):  # pylint: disable=redefined-builtin
        with self._lock:
            if all:
                sql, args = "SELECT COUNT(*) FROM entries", ()
            else:
                sql, args = "SELECT COUNT(*) FROM entries WHERE state = ?", (state,)
            return int(self._db.execute(sql, args).fetchone()[0])

    # -- tier 2 ------------------------------------------------------------------
    def compact(self):
        """Hard-delete the marked rows and rebuild the index from what is left.

        Because the index is derived from sqlite, compaction IS the rebuild, and
        ids never move: they are sqlite rowids, not index positions.
        """
        with self._lock:
            self._db.execute("DELETE FROM entries WHERE state = -1")
            self._db.commit()
            self._load()
            return True

    def sweep_expired(self):
        """Soft-delete rows past the store's default max_age. Returns the count.

        Run from maybe_compact rather than from fetch: expiring on read would
        make get() a writer, and it would only ever reclaim rows somebody
        happened to look up. A sweep also catches the ones nobody reads, which
        is most of what goes stale.
        """
        with self._lock:
            if self._max_age is None:
                return 0
            cur = self._db.execute(
                "UPDATE entries SET state = -1 WHERE state = 0 AND created_at < ?",
                (self._now() - self._max_age,),
            )
            self._db.commit()
            for row_id in [r[0] for r in self._db.execute(
                    "SELECT id FROM entries WHERE state = -1")]:
                self._lru.pop(int(row_id), None)
            return cur.rowcount

    def maybe_compact(self):
        """Compact if the marked rows are worth reclaiming. Called from put --
        the only place that grows the store.

        Once the cache is full every put evicts one entry, so the rate trigger
        fires roughly every max_size/9 writes: ~1 in 118 at the default 1000,
        and on nearly every write for a very small cache. That is more often
        than a naive reading suggests, but measured cost is
        ~17us/put amortized at max_size=1000, which stays invisible next to the
        LLM call being avoided. Raise max_mark_rate if you ever see it.
        """
        with self._lock:
            self.sweep_expired()
            if not should_compact(
                self.count(state=-1), self.count(all=True),
                self._max_mark_count, self._max_mark_rate,
            ):
                return False
            return self.compact()

    def close(self):
        with self._lock:
            self._db.commit()
            self._db.close()
