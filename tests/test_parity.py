"""Does semcache make the same hit/miss decision as GPTCache?

The point is attribution, not correctness: BRIEF-level analysis showed no cosine
threshold separates the golden sets, and until this test exists nobody knows
whether that is inherited from the embedder or introduced by the rebuild.
Neither cache is asserted to be right -- only that they agree.

The threshold equivalence, verified on a live pair before the run:

    SearchDistanceEvaluation(max_distance=4.0, positive=False)
        score          = 4.0 - L2
        rank_threshold = (max_rank - min_rank) * similarity_threshold
                       = 4.0 * 0.8 = 3.2
        hit  <=>  4.0 - L2 >= 3.2  <=>  L2 <= 0.8

    and for L2-normalized vectors  L2^2 = 2 - 2cos, so
        L2 <= 0.8  <=>  cos >= 1 - 0.8^2/2 = 0.68

Requires gptcache, which needs its own venv (it pulls sqlalchemy, faiss and a
large tree, and conflicts with the main one):

    python -m venv .venv-parity
    .venv-parity/bin/pip install gptcache sqlalchemy faiss-cpu \
                                 sentence-transformers pytest
    OMP_NUM_THREADS=1 PYTHONPATH=. .venv-parity/bin/python \
        -m pytest tests/test_parity.py -q -m slow

OMP_NUM_THREADS=1 is required: faiss and torch both load OpenMP and the
combination segfaults (exit 139) on Python 3.14 without it.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # must precede faiss/torch

import tempfile                                  # noqa: E402

import numpy as np                               # noqa: E402
import pytest                                    # noqa: E402

from .golden_pairs import MUST_HIT, MUST_MISS    # noqa: E402

pytestmark = pytest.mark.slow

gptcache = pytest.importorskip("gptcache",
                               reason="parity needs gptcache in .venv-parity")
pytest.importorskip("faiss")
pytest.importorskip("sqlalchemy")

PAIRS = [(p.a, p.b, p.why) for p in MUST_HIT + MUST_MISS]


@pytest.fixture(scope="module")
def embed():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def fn(text, **_):
        v = model.encode(text).astype("float32")
        return v / np.linalg.norm(v)

    return fn


@pytest.fixture(scope="module")
def gpt_cache(embed):
    from gptcache import Cache, Config
    from gptcache.manager import CacheBase, VectorBase, get_data_manager
    from gptcache.processor.pre import get_prompt
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

    d = tempfile.mkdtemp()
    c = Cache()
    c.init(
        pre_embedding_func=get_prompt,
        embedding_func=embed,
        data_manager=get_data_manager(
            CacheBase("sqlite", sql_url="sqlite:///" + d + "/g.db"),
            VectorBase("faiss", dimension=384, index_path=d + "/g.index"),
        ),
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=0.8),
    )
    return c


@pytest.fixture(scope="module")
def sem_cache(embed):
    from semcache import Cache
    from semcache.cache import GPTCACHE_EQUIVALENT_THRESHOLD

    # The raw cosine path, so this compares the retrieval decision rather than
    # semcache's extra guards: no ID skip, no reranker, no verifier.
    # 0.60, not semcache's shipped 0.68: this arm exists to reproduce
    # gptcache's decisions, not semcache's defaults. See the module docstring.
    c = Cache(":memory:", embedder=lambda t: embed(t), dim=384,
              threshold=GPTCACHE_EQUIVALENT_THRESHOLD, skip_pattern=None,
              max_size=10_000)
    yield c
    c.close()


def test_threshold_mapping_holds_on_a_known_pair(gpt_cache, sem_cache, embed):
    """Verify the L2 <-> cosine equivalence before trusting the whole run."""
    from gptcache.adapter.api import get as gget, put as gput

    a, b = "what is github?", "What is GitHub"
    gput(a, "X", cache_obj=gpt_cache)
    sem_cache.put(a, "X")
    l2 = float(np.linalg.norm(embed(a) - embed(b)))
    cos = float(np.dot(embed(a), embed(b)))
    assert (l2 ** 2 <= 0.8) == (cos >= 0.60), "the algebraic mapping is wrong"
    assert (gget(b, cache_obj=gpt_cache) == "X") == (l2 ** 2 <= 0.8)
    assert (sem_cache.get(b) == "X") == (cos >= 0.60)


@pytest.mark.parametrize("a,b,why", PAIRS, ids=[f"{p[2]}" for p in PAIRS])
def test_same_decision_as_gptcache(a, b, why, gpt_cache, sem_cache):
    from gptcache.adapter.api import get as gget, put as gput

    gput(a, "X", cache_obj=gpt_cache)
    sem_cache.put(a, "X")
    gpt_hit = gget(b, cache_obj=gpt_cache) == "X"
    sem_hit = sem_cache.get(b) == "X"
    assert gpt_hit == sem_hit, (
        f"divergent decision on [{why}] {a!r} / {b!r}: "
        f"gptcache={'HIT' if gpt_hit else 'MISS'}, "
        f"semcache={'HIT' if sem_hit else 'MISS'}"
    )
