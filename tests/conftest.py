"""Shared fixtures. Layers 1 and 4 test the data structure, not the model, so
they run against a deterministic stub embedder and never download anything.
"""

import zlib

import numpy as np
import pytest

from semcache import Cache
from semcache.store import Store

DIM = 8


def stub(text, dim=DIM):
    """Deterministic embedding. crc32, not hash() -- hash() is salted per process."""
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    v = rng.standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)


def vec(i, dim=DIM):
    """A distinct unit vector per integer, stable across runs."""
    return stub(f"vec-{i}", dim)


def at_cosine(c, dim=DIM):
    """Two unit vectors whose dot product is exactly `c`. For threshold tests."""
    a = np.zeros(dim, dtype="float32")
    a[0] = 1.0
    b = np.zeros(dim, dtype="float32")
    b[0], b[1] = c, np.sqrt(1.0 - c * c)
    return a, b


@pytest.fixture
def store():
    s = Store(":memory:", dim=DIM, max_size=100_000)
    yield s
    s.close()


@pytest.fixture
def store_maxsize_10():
    s = Store(":memory:", dim=DIM, max_size=10)
    yield s
    s.close()


@pytest.fixture
def cache():
    c = Cache(":memory:", embedder=stub, dim=DIM)
    yield c
    c.close()


class Clock:
    """Injectable time source. Tests must never sleep."""

    def __init__(self, t=1_000_000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()
