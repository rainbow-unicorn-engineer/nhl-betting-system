"""
Tests for ingestion/polite.py: the gap between request starts, retries
on timeouts / 429 / 5xx (with Retry-After), no retry on other 4xx, and
404 as an answer. No network: the session is a stub, the clock is fake.
"""
import requests

from ingestion.polite import PoliteClient


class FakeResp:
    def __init__(self, status, body=None, headers=None, bad_json=False):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.t += s


def client(replies, **kw):
    clock = Clock()
    c = PoliteClient("test", session=FakeSession(replies), clock=clock.now,
                     sleep=clock.sleep, **kw)
    return c, clock


def test_ok_returns_the_parsed_body():
    c, _ = client([FakeResp(200, {"a": 1})])
    r = c.get_json("u", {"x": 1})
    assert (r.status, r.body, r.http_status) == ("ok", {"a": 1}, 200)
    assert c.session.calls == [("u", {"x": 1})]


def test_requests_are_spaced_by_the_minimum_gap():
    c, clock = client([FakeResp(200, {}), FakeResp(200, {}), FakeResp(200, {})],
                      min_interval_s=0.5)
    for _ in range(3):
        c.get_json("u")
    assert clock.sleeps == [0.5, 0.5]       # the first request does not wait
    assert c.requests == 3


def test_404_is_an_answer_not_retried():
    c, _ = client([FakeResp(404)])
    r = c.get_json("u")
    assert r.status == "not_found" and len(c.session.calls) == 1


def test_other_4xx_is_an_error_not_retried():
    c, _ = client([FakeResp(403)])
    r = c.get_json("u")
    assert (r.status, r.http_status, r.problem) == ("error", 403, "HTTP 403")
    assert len(c.session.calls) == 1


def test_429_honours_retry_after_then_succeeds():
    c, clock = client([FakeResp(429, headers={"Retry-After": "7"}), FakeResp(200, [1])],
                      min_interval_s=0.0)
    r = c.get_json("u")
    assert r.status == "ok" and r.body == [1]
    assert 7.0 in clock.sleeps


def test_5xx_timeouts_and_bad_json_retry_with_growing_pause_then_give_up():
    c, clock = client([FakeResp(503), requests.Timeout(), FakeResp(200, bad_json=True),
                       requests.ConnectionError()], min_interval_s=0.0, retries=4,
                      backoff_s=2.0)
    r = c.get_json("u")
    assert r.status == "error" and "connection failed" in r.problem
    assert clock.sleeps == [2.0, 4.0, 6.0]
    assert len(c.session.calls) == 4


def test_unparseable_retry_after_falls_back_to_backoff():
    c, clock = client([FakeResp(429, headers={"Retry-After": "soon"}), FakeResp(200, {})],
                      min_interval_s=0.0, backoff_s=3.0)
    assert c.get_json("u").status == "ok"
    assert clock.sleeps == [3.0]
