"""Drop-in replacement for Cache that talks to a semcache server.

Same get/put signatures, so swapping one for the other is a one-line change.
"""

import requests


class Client:
    def __init__(self, url, timeout=10):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def get(self, prompt, threshold=None, session_id=None, max_age=None):
        body = {"prompt": prompt, "threshold": threshold,
                "session_id": session_id, "max_age": max_age}
        resp = requests.post(f"{self.url}/get", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["answer"]

    def put(self, prompt, answer, session_id=None):
        body = {"prompt": prompt, "answer": answer, "session_id": session_id}
        resp = requests.post(f"{self.url}/put", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["id"]

    def health(self):
        resp = requests.get(f"{self.url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
