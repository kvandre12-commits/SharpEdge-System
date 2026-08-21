from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_read_view import render_live_read_html
from post_apple_rotation import build_post_apple_rotation_live
from post_apple_rotation_view import render_post_apple_rotation_block


def _write_study(path: Path) -> None:
    payload = {
        "schema": "sharpedge.apple_earnings_reaction_dips.v1",
        "generated_at_utc": "2026-08-02T00:00:00+00:00",
        "assumption": "AAPL reports after close; reaction day is next trading session.",
        "symbols": ["AAPL", "AMD", "AMZN", "NVDA", "TSLA", "QQQ", "SPY"],
        "summaries": {
            "AAPL": {"reaction_opened_below_prior_close": {"count": 18}},
            "AMD": {
                "reaction_opened_below_prior_close": {
                    "count": 18,
                    "median_return_5d_pct": 6.3,
                    "return_5d_positive_pct": 72.2,
                    "median_return_3d_pct": 4.1,
                    "median_consecutive_lower_opens_from_reaction": 2.5,
                }
            },
            "AMZN": {
                "reaction_opened_below_prior_close": {
                    "count": 15,
                    "median_return_5d_pct": 0.6,
                    "return_5d_positive_pct": 53.3,
                    "median_return_3d_pct": 0.8,
                    "median_consecutive_lower_opens_from_reaction": 2.0,
                }
            },
            "NVDA": {
                "reaction_opened_below_prior_close": {
                    "count": 19,
                    "median_return_5d_pct": 2.3,
                    "return_5d_positive_pct": 68.4,
                    "median_return_3d_pct": 3.3,
                    "median_consecutive_lower_opens_from_reaction": 3.0,
                }
            },
            "TSLA": {
                "reaction_opened_below_prior_close": {
                    "count": 16,
                    "median_return_5d_pct": 3.3,
                    "return_5d_positive_pct": 75.0,
                    "median_return_3d_pct": 2.1,
                    "median_consecutive_lower_opens_from_reaction": 2.0,
                }
            },
            "QQQ": {
                "reaction_opened_below_prior_close": {
                    "count": 13,
                    "median_return_5d_pct": 0.4,
                    "return_5d_positive_pct": 53.8,
                    "median_return_3d_pct": 0.2,
                    "median_consecutive_lower_opens_from_reaction": 2.0,
                }
            },
            "SPY": {
                "reaction_opened_below_prior_close": {
                    "count": 13,
                    "median_return_5d_pct": 0.0,
                    "return_5d_positive_pct": 53.8,
                    "median_return_3d_pct": 0.0,
                    "median_consecutive_lower_opens_from_reaction": 2.0,
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows(
    open_price: float,
    close_price: float,
    high: float,
    low: float,
    session_date: str,
) -> list[dict]:
    return [
        {
            "date": session_date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
            "volume": 100,
        },
        {
            "date": session_date,
            "open": close_price,
            "high": high,
            "low": low,
            "close": close_price,
            "volume": 120,
        },
    ]


def _session_fetcher_factory(snapshot_map: dict[str, dict[str, float]]):
    def _fetcher(symbol: str):
        snap = snapshot_map[symbol]
        return _rows(
            snap["open"],
            snap["close"],
            snap["high"],
            snap["low"],
            snap.get("session_date", "2026-08-03"),
        ), {
            "chart_previous_close": snap["previous_close"],
            "market_state": "REGULAR",
            "last_bar_utc": "2026-08-03T15:31:00+00:00",
        }

    return _fetcher


def _daily_fetcher_factory(rows_by_symbol: dict[str, list[dict]]):
    def _fetcher(symbol: str):
        return rows_by_symbol[symbol], {"provider": "test-daily"}

    return _fetcher


def test_post_apple_rotation_requires_verified_lower_open_window(tmp_path):
    study_path = tmp_path / "apple.json"
    _write_study(study_path)
    session_fetcher = _session_fetcher_factory(
        {
            "AAPL": {
                "previous_close": 100,
                "open": 98,
                "close": 97.5,
                "high": 98.2,
                "low": 97.0,
            },
            "AMD": {
                "previous_close": 100,
                "open": 100.0,
                "close": 101.8,
                "high": 102.0,
                "low": 99.9,
            },
            "AMZN": {
                "previous_close": 100,
                "open": 100.2,
                "close": 100.5,
                "high": 100.7,
                "low": 100.0,
            },
            "NVDA": {
                "previous_close": 100,
                "open": 99.7,
                "close": 101.0,
                "high": 101.2,
                "low": 99.5,
            },
            "TSLA": {
                "previous_close": 100,
                "open": 100.4,
                "close": 103.0,
                "high": 103.3,
                "low": 100.3,
            },
            "QQQ": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.4,
                "high": 100.5,
                "low": 99.9,
            },
            "SPY": {
                "previous_close": 100,
                "open": 99.9,
                "close": 100.2,
                "high": 100.3,
                "low": 99.8,
            },
        }
    )
    daily_fetcher = _daily_fetcher_factory(
        {
            "AAPL": [
                {
                    "date": "2026-07-30",
                    "open": 101.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 100.0,
                },
                {
                    "date": "2026-07-31",
                    "open": 97.0,
                    "high": 98.0,
                    "low": 95.5,
                    "close": 96.2,
                },
                {
                    "date": "2026-08-03",
                    "open": 98.0,
                    "high": 99.0,
                    "low": 97.5,
                    "close": 97.5,
                },
            ]
        }
    )

    packet = build_post_apple_rotation_live(
        study_path=study_path,
        fetcher=session_fetcher,
        daily_fetcher=daily_fetcher,
        earnings_dates=["2026-07-30"],
    )

    assert packet["mode"] == "trade_today"
    assert packet["verified_window"]["active"] is True
    assert packet["verified_window"]["sessions_since_reaction"] == 1
    assert [row["symbol"] for row in packet["today_trades"]] == ["TSLA", "AMD", "NVDA"]
    assert all(row["role"] == "leader" for row in packet["today_trades"])
    assert packet["today_trades"][0]["lane_label"] in {
        "GO WITH STRENGTH",
        "LONG ONLY ON PULLBACK / RECLAIM",
    }


def test_post_apple_rotation_stands_down_outside_verified_window(tmp_path):
    study_path = tmp_path / "apple.json"
    _write_study(study_path)
    session_fetcher = _session_fetcher_factory(
        {
            "AAPL": {
                "previous_close": 100,
                "open": 99.8,
                "close": 100.1,
                "high": 100.2,
                "low": 99.6,
                "session_date": "2026-08-10",
            },
            "AMD": {
                "previous_close": 100,
                "open": 100.0,
                "close": 101.0,
                "high": 101.1,
                "low": 99.8,
                "session_date": "2026-08-10",
            },
            "AMZN": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.3,
                "high": 100.4,
                "low": 99.8,
                "session_date": "2026-08-10",
            },
            "NVDA": {
                "previous_close": 100,
                "open": 99.9,
                "close": 100.2,
                "high": 100.3,
                "low": 99.7,
                "session_date": "2026-08-10",
            },
            "TSLA": {
                "previous_close": 100,
                "open": 100.1,
                "close": 100.4,
                "high": 100.6,
                "low": 100.0,
                "session_date": "2026-08-10",
            },
            "QQQ": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.2,
                "high": 100.3,
                "low": 99.9,
                "session_date": "2026-08-10",
            },
            "SPY": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.2,
                "high": 100.3,
                "low": 99.9,
                "session_date": "2026-08-10",
            },
        }
    )
    daily_fetcher = _daily_fetcher_factory(
        {
            "AAPL": [
                {
                    "date": "2026-07-30",
                    "open": 101.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 100.0,
                },
                {
                    "date": "2026-07-31",
                    "open": 97.0,
                    "high": 98.0,
                    "low": 95.5,
                    "close": 96.2,
                },
                {
                    "date": "2026-08-03",
                    "open": 98.0,
                    "high": 99.0,
                    "low": 97.5,
                    "close": 97.8,
                },
                {
                    "date": "2026-08-04",
                    "open": 98.2,
                    "high": 99.2,
                    "low": 97.9,
                    "close": 98.5,
                },
                {
                    "date": "2026-08-05",
                    "open": 98.8,
                    "high": 99.5,
                    "low": 98.1,
                    "close": 98.9,
                },
                {
                    "date": "2026-08-06",
                    "open": 99.1,
                    "high": 99.7,
                    "low": 98.7,
                    "close": 99.0,
                },
                {
                    "date": "2026-08-07",
                    "open": 99.2,
                    "high": 100.0,
                    "low": 98.9,
                    "close": 99.4,
                },
                {
                    "date": "2026-08-10",
                    "open": 99.4,
                    "high": 100.1,
                    "low": 99.0,
                    "close": 100.1,
                },
            ]
        }
    )

    packet = build_post_apple_rotation_live(
        study_path=study_path,
        fetcher=session_fetcher,
        daily_fetcher=daily_fetcher,
        earnings_dates=["2026-07-30"],
    )

    assert packet["mode"] == "inactive_window"
    assert packet["verified_window"]["active"] is False
    assert packet["today_trades"] == []


def test_post_apple_rotation_renders_into_live_read_html(tmp_path):
    study_path = tmp_path / "apple.json"
    _write_study(study_path)
    session_fetcher = _session_fetcher_factory(
        {
            "AAPL": {
                "previous_close": 100,
                "open": 98,
                "close": 97.0,
                "high": 98.1,
                "low": 96.8,
            },
            "AMD": {
                "previous_close": 100,
                "open": 100.0,
                "close": 101.5,
                "high": 101.7,
                "low": 99.9,
            },
            "AMZN": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.1,
                "high": 100.2,
                "low": 99.8,
            },
            "NVDA": {
                "previous_close": 100,
                "open": 100.0,
                "close": 101.1,
                "high": 101.2,
                "low": 99.9,
            },
            "TSLA": {
                "previous_close": 100,
                "open": 100.4,
                "close": 103.2,
                "high": 103.5,
                "low": 100.3,
            },
            "QQQ": {
                "previous_close": 100,
                "open": 100.0,
                "close": 100.3,
                "high": 100.4,
                "low": 99.9,
            },
            "SPY": {
                "previous_close": 100,
                "open": 99.9,
                "close": 100.2,
                "high": 100.3,
                "low": 99.8,
            },
        }
    )
    daily_fetcher = _daily_fetcher_factory(
        {
            "AAPL": [
                {
                    "date": "2026-07-30",
                    "open": 101.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 100.0,
                },
                {
                    "date": "2026-07-31",
                    "open": 97.0,
                    "high": 98.0,
                    "low": 95.5,
                    "close": 96.2,
                },
                {
                    "date": "2026-08-03",
                    "open": 98.0,
                    "high": 99.0,
                    "low": 97.5,
                    "close": 97.0,
                },
            ]
        }
    )
    packet = build_post_apple_rotation_live(
        study_path=study_path,
        fetcher=session_fetcher,
        daily_fetcher=daily_fetcher,
        earnings_dates=["2026-07-30"],
    )

    block = render_post_apple_rotation_block(packet)
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": 0.3, "vwap": 99.8},
        op={"put_wall": 98.5, "call_wall": 101.5},
        lines=[("Rotation", "info", "watch leadership")],
        post_apple_rotation=packet,
        stamp="12:00:00",
    )

    assert "POST-AAPL ROTATION CARD" in block
    assert "TRADE TODAY" in block
    assert "Verified post-AAPL dip day" in block
    assert "GO WITH STRENGTH" in block or "LONG ONLY ON PULLBACK / RECLAIM" in block
    assert "POST-AAPL ROTATION CARD" in html
    assert "sessions since reaction" in html
