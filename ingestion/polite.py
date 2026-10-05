"""
ingestion/polite.py
A small, polite HTTP client for the free data loaders (ingestion/nhl_shifts.py,
ingestion/nhl_game_info.py, ingestion/kalshi.py).

"Polite" means three things:
- a minimum gap between the START of two requests (min_interval_s), so a
  loader never asks a free service faster than it allows: 0.34 s is at
  most about 3 requests a second however fast the replies come back;
- a retry with a growing pause on a timeout, a dropped connection, HTTP 429
  ("too many requests", honouring the server's Retry-After when it sends
  one) and HTTP 5xx (a server-side failure);
- no retry on any other 4xx: 404 ("not found") is an answer, not a fault.

get_json() never raises for a network problem. It returns a Reply whose
status says what happened: "ok" (body is the parsed JSON), "not_found"
(HTTP 404), or "error" (anything else, after the retries; `problem` says
what). The loaders write that status into their fetch logs, so a re-run
retries only what failed.
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

logger = logging.getLogger("nhl.ingestion.polite")

RETRY_STATUSES = (429, 500, 502, 503, 504)


@dataclass
class Reply:
    status: str                 # ok, not_found, error
    body: Optional[object] = None
    http_status: Optional[int] = None
    problem: Optional[str] = None


class PoliteClient:
    """requests.Session with a minimum gap between request starts and
    retries. `clock` and `sleep` are injectable so tests run instantly."""

    def __init__(self, name: str, min_interval_s: float = 0.34, retries: int = 4,
                 timeout_s: float = 30.0, backoff_s: float = 2.0,
                 session: Optional[requests.Session] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 headers: Optional[dict] = None):
        self.name = name
        self.min_interval_s = min_interval_s
        self.retries = max(1, retries)
        self.timeout_s = timeout_s
        self.backoff_s = backoff_s
        self.session = session or requests.Session()
        if headers:
            self.session.headers.update(headers)
        self._clock, self._sleep = clock, sleep
        self._last_start: Optional[float] = None
        self.requests = 0

    def _wait_turn(self) -> None:
        if self._last_start is not None:
            gap = self.min_interval_s - (self._clock() - self._last_start)
            if gap > 0:
                self._sleep(gap)
        self._last_start = self._clock()

    def _retry_after(self, resp) -> Optional[float]:
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            return min(float(value), 120.0) if value is not None else None
        except (TypeError, ValueError):
            return None

    def get_json(self, url: str, params: Optional[dict] = None) -> Reply:
        problem, http_status = "no attempt made", None
        for attempt in range(1, self.retries + 1):
            self._wait_turn()
            self.requests += 1
            wait = None
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_s)
                http_status = resp.status_code
                if resp.status_code == 404:
                    return Reply("not_found", None, 404, "HTTP 404")
                if resp.status_code in RETRY_STATUSES:
                    problem = f"HTTP {resp.status_code}"
                    wait = self._retry_after(resp)
                elif resp.status_code >= 400:
                    return Reply("error", None, resp.status_code,
                                 f"HTTP {resp.status_code}")
                else:
                    try:
                        return Reply("ok", resp.json(), resp.status_code)
                    except ValueError:
                        problem = "response is not JSON"
            except requests.Timeout:
                problem = f"timed out after {self.timeout_s:.0f}s"
            except requests.ConnectionError as e:
                problem = f"connection failed ({type(e).__name__})"
            except requests.RequestException as e:
                problem = f"request failed ({type(e).__name__})"
            if attempt < self.retries:
                pause = wait if wait is not None else self.backoff_s * attempt
                logger.warning(f"{self.name}: {problem}; retry {attempt}/{self.retries - 1} "
                               f"in {pause:.0f}s")
                self._sleep(pause)
        logger.error(f"{self.name}: giving up after {self.retries} attempts: {problem}")
        return Reply("error", None, http_status, problem)
