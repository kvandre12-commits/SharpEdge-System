from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import http_utils  # noqa: E402
import market_data_sources as mds  # noqa: E402


class DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def test_fetch_yahoo_chart_result_retries_on_429_then_succeeds():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD"},
                    "timestamp": [1710000000],
                    "indicators": {"quote": [{"close": [100.0], "volume": [10]}]},
                }
            ]
        }
    }
    responses = [
        DummyResponse(429, {}),
        DummyResponse(429, {}),
        DummyResponse(200, payload),
    ]

    with (
        patch.object(http_utils.requests, "get", side_effect=responses) as mocked_get,
        patch.object(http_utils.time, "sleep", return_value=None),
    ):
        result = mds.fetch_yahoo_chart_result(interval="1m", range_="1d", timeout=5)

    assert result["meta"]["currency"] == "USD"
    assert mocked_get.call_count == 3


def test_drop_terminal_placeholder_rows_preserves_real_rows():
    rows = [
        (0, 100.0, 100.5, 99.9, 100.2, 1000),
        (1, 100.2, 100.2, 100.2, 100.2, 0),
    ]

    clean, dropped = mds._drop_terminal_placeholder_rows(rows)

    assert clean == [rows[0]]
    assert dropped == 1


def test_drop_terminal_placeholder_rows_keeps_only_row_and_nonflat_zero_volume():
    single = [(0, 100.0, 100.0, 100.0, 100.0, 0)]
    nonflat = [
        (0, 100.0, 100.5, 99.9, 100.2, 1000),
        (1, 100.2, 100.4, 100.1, 100.3, 0),
    ]

    assert mds._drop_terminal_placeholder_rows(single) == (single, 0)
    assert mds._drop_terminal_placeholder_rows(nonflat) == (nonflat, 0)


def test_fetch_cboe_options_book_returns_richer_source_fields():
    payload = {
        "data": {
            "current_price": 101.5,
            "close": 101.2,
            "open": 100.9,
            "high": 102.0,
            "low": 100.5,
            "prev_day_close": 100.8,
            "bid": 101.4,
            "ask": 101.6,
            "bid_size": 10,
            "ask_size": 12,
            "price_change": 0.7,
            "price_change_percent": 0.69,
            "iv30": 15.7,
            "iv30_change": 0.3,
            "iv30_change_percent": 1.9,
            "last_trade_time": "2026-06-25T14:01:26",
            "options": [
                {
                    "option": "SPY260625C00101000",
                    "last_trade_time": "2026-06-25T14:01:26",
                }
            ],
        }
    }

    with patch.object(mds, "request_json_with_backoff", return_value=payload):
        spot, _book, source = mds.fetch_cboe_options_book(timeout=5)

    assert spot == 101.5
    assert source["open"] == 100.9
    assert source["high"] == 102.0
    assert source["low"] == 100.5
    assert source["prev_day_close"] == 100.8
    assert source["bid"] == 101.4
    assert source["ask"] == 101.6
    assert source["price_change_percent"] == 0.69
    assert source["iv30_change_percent"] == 1.9
    assert source["latest_option_trade_time_raw"] == "2026-06-25T14:01:26"
