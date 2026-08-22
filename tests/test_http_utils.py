from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import http_utils as hu


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"status {self.status_code}")
            err.response = self
            raise err


def test_request_json_returns_parsed_payload():
    with patch.object(hu.requests, "get", return_value=FakeResponse(json_data={"a": 1})):
        assert hu.request_json_with_backoff("http://x") == {"a": 1}


def test_request_text_returns_body():
    with patch.object(hu.requests, "get", return_value=FakeResponse(text="<html>ok</html>")):
        assert hu.request_text_with_backoff("http://x") == "<html>ok</html>"


def test_retries_on_429_then_succeeds():
    responses = [FakeResponse(status_code=429), FakeResponse(json_data={"ok": True})]
    with patch.object(hu.requests, "get", side_effect=responses) as mock_get:
        result = hu.request_json_with_backoff("http://x", base_sleep_seconds=0)
    assert result == {"ok": True}
    assert mock_get.call_count == 2


def test_non_retryable_status_raises():
    with patch.object(hu.requests, "get", return_value=FakeResponse(status_code=403)):
        try:
            hu.request_text_with_backoff("http://x", base_sleep_seconds=0)
        except requests.HTTPError as exc:
            assert exc.response.status_code == 403
        else:
            raise AssertionError("expected HTTPError for non-retryable 403")
