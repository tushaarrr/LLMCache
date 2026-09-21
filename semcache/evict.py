"""The two eviction tiers.

Tier 1, soft delete: the LRU is full, it pops the least-recently-used id, the
row is marked state = -1. Cheap and frequent -- deleting from a vector index is
expensive, marking a sqlite row is not.

Tier 2, compaction: once the marked rows are worth reclaiming, hard-delete them
and rebuild the index from what is left. Expensive and rare.
"""

import cachetools

# The original's numbers (EvictionManager.MAX_MARK_COUNT / MAX_MARK_RATE).
# Reasonable defaults, not physics -- both are constructor args on Store.
MAX_MARK_COUNT = 5000
MAX_MARK_RATE = 0.1


class LRU(cachetools.LRUCache):
    """cachetools has no eviction hook, so override popitem.

    Never call .clear() on this. cachetools' Cache.clear() empties the backing
    dict directly and does NOT go through popitem, so on_evict never fires: the
    ids are simply forgotten while their rows stay live in sqlite, unevictable
    and unbounded by max_size. To reset it, build a new instance.
    """

    def __init__(self, maxsize, on_evict):
        super().__init__(maxsize)
        self.on_evict = on_evict

    def popitem(self):
        key, value = super().popitem()
        self.on_evict([key])
        return key, value


def should_compact(marked, total, max_count=MAX_MARK_COUNT, max_rate=MAX_MARK_RATE):
    """Is reclaiming the marked rows worth a full index rebuild yet?"""
    if marked <= 0:
        return False
    if marked >= max_count:
        return True
    return total > 0 and marked / total >= max_rate
