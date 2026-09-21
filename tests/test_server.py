"""The HTTP front end. Real transport throughout -- a mocked transport tests
nothing about the gap this step closes.

Runs against a real uvicorn subprocess so the client's requests calls, the
JSON contract and the process lifecycle are all exercised together.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("requests")

from semcache import Cache                       # noqa: E402
from semcache.client import Client               # noqa: E402

from .conftest import DIM, stub                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, url, proc):
        self.url, self.proc = url, proc


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = _free_port()
    db = tmp_path_factory.mktemp("srv") / "c.db"
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "semcache.server", "--db", str(db),
         "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    import requests
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server died:\n{proc.stdout.read().decode()}")
        try:
            if requests.get(f"{url}/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("server never became healthy")
    yield Server(url, proc)
    proc.terminate()
    proc.wait(timeout=10)


def test_health(server):
    body = Client(server.url).health()
    assert body["status"] == "ok"
    assert isinstance(body["entries"], int)


def test_two_clients_share_one_cache(server):
    """THE point of this step: in-process, 4 gunicorn workers are 4 caches."""
    a, b = Client(server.url), Client(server.url)
    a.put("what is git", "version control")
    assert b.get("what's git") == "version control"


def test_put_returns_an_id_and_get_misses_cleanly(server):
    c = Client(server.url)
    assert isinstance(c.put("an isolated prompt about llamas", "yes"), int)
    assert c.get("something with no relation whatsoever to that") is None


def test_client_and_cache_are_interchangeable(server, tmp_path):
    """Same call sequence against both, identical results."""
    remote = Client(server.url)
    local = Cache(tmp_path / "local.db")

    calls = [("put", "how do I center a div", "use flexbox"),
             ("get", "how do I center a div", None),
             ("get", "what is the capital of France", None),
             ("put", "what is rust", "a systems language"),
             ("get", "what is rust", None)]
    out = {"remote": [], "local": []}
    for name, target in (("remote", remote), ("local", local)):
        for op, prompt, answer in calls:
            if op == "put":
                out[name].append(isinstance(target.put(prompt, answer), int))
            else:
                out[name].append(target.get(prompt))
    local.close()
    assert out["remote"] == out["local"]


def test_sessions_survive_the_wire(server):
    c = Client(server.url)
    c.put("what is our revenue", "42M", session_id="acme")
    assert c.get("what is our revenue", session_id="globex") is None
    assert c.get("what is our revenue", session_id="acme") == "42M"


def test_server_survives_a_bad_request(server):
    """Missing field -> 4xx, not 500, and the process stays up."""
    import requests

    r = requests.post(f"{server.url}/put", json={"prompt": "no answer field"},
                      timeout=10)
    assert 400 <= r.status_code < 500, f"got {r.status_code}, expected 4xx"

    r = requests.post(f"{server.url}/get", json={}, timeout=10)
    assert 400 <= r.status_code < 500

    r = requests.post(f"{server.url}/get", data=b"not json", timeout=10)
    assert 400 <= r.status_code < 500

    assert Client(server.url).health()["status"] == "ok"   # still alive


def test_identifier_guard_applies_over_http(server):
    c = Client(server.url)
    assert c.put("status of ticket ENG-9911", "closed") is None
    assert c.get("status of ticket ENG-9911") is None
