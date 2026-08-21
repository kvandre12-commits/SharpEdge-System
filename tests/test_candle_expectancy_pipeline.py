from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_candle_expectancy_events import COLUMNS  # noqa: E402
from build_candle_conditioned_expectancy_matrix import build_matrix  # noqa: E402
from build_candle_confidence_weights import build_confidence  # noqa: E402
from candle_expectancy_core import (  # noqa: E402
    STOP_FIRST,
    TARGET_FIRST,
    build_event_rows_for_session,
    classify_event,
    first_touch_outcome,
)


def _bar(
    idx: int, open_: float, high: float, low: float, close: float, volume: int = 1000
):
    return {
        "ts": f"2026-01-02T{9 + (30 + idx * 15) // 60:02d}:{(30 + idx * 15) % 60:02d}:00-05:00",
        "session_date": "2026-01-02",
        "symbol": "SPY",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "trade_count": volume // 10,
        "vwap": close - 0.02,
    }


def test_classify_event_detects_inside_bar_before_single_candle_noise():
    previous = _bar(0, 100.0, 101.0, 99.0, 100.5)
    current = _bar(1, 100.4, 100.8, 99.4, 100.6)

    event = classify_event(current, previous)

    assert event == {"event_name": "inside_bar", "event_direction": "NEUTRAL"}


def test_first_touch_outcome_orders_target_before_stop_for_calls_and_puts():
    calls = first_touch_outcome(
        100.0,
        [_bar(1, 100.0, 100.2, 99.95, 100.1)],
        "CALLS",
        target_pct=0.001,
        stop_pct=0.001,
    )
    puts = first_touch_outcome(
        100.0,
        [_bar(1, 100.0, 100.05, 99.8, 99.9)],
        "PUTS",
        target_pct=0.001,
        stop_pct=0.001,
    )
    stopped = first_touch_outcome(
        100.0,
        [_bar(1, 100.0, 100.05, 99.8, 99.9)],
        "CALLS",
        target_pct=0.001,
        stop_pct=0.001,
    )

    assert calls["target_before_stop_label"] == TARGET_FIRST
    assert puts["target_before_stop_label"] == TARGET_FIRST
    assert stopped["target_before_stop_label"] == STOP_FIRST


def test_build_event_rows_attach_causal_context_and_forward_outcome():
    bars = [
        _bar(0, 100.0, 100.2, 99.8, 100.1, 1000),
        _bar(1, 100.1, 100.3, 99.9, 100.0, 900),
        _bar(2, 100.0, 100.05, 99.4, 99.95, 1800),
        _bar(3, 99.95, 100.3, 99.9, 100.25, 2500),
        _bar(4, 100.25, 100.5, 100.2, 100.45, 2600),
    ]

    rows = build_event_rows_for_session(
        symbol="SPY",
        session_date="2026-01-02",
        bars=bars,
        daily={"prior_high": 100.2, "prior_low": 98.8, "prior_close": 99.5},
        regime={"vol_state": "LOW_VOL", "macro_state": "RISK_ON", "dp_state": "NORMAL"},
        open_regime={"open_regime_label": "OPEN_DRIVE", "setup_dir": "UP"},
        options={"gamma_wall_strike": 101.0, "pcr_oi": 0.9},
        horizon_bars=2,
        target_pct=0.001,
        stop_pct=0.001,
    )

    event_names = {row["event_name"] for row in rows}
    assert event_names & {"bearish_engulfing", "bearish_conviction", "supply_tail"}
    assert any(
        row["nearest_reference_name"] in {"PDH", "VWAP", "ORL_30m", "ORH_30m"}
        for row in rows
    )
    assert any(
        row["volume_confirmation"] in {"confirmed", "participating", "mixed"}
        for row in rows
    )
    assert all(row["vol_state"] == "LOW_VOL" for row in rows)
    assert all(row["target_before_stop_label"] for row in rows)


def test_build_candle_conditioned_matrix_aggregates_causal_groups():
    con = sqlite3.connect(":memory:")
    names = [name for name, _ in COLUMNS]
    ddl = ", ".join(f"{name} {typ}" for name, typ in COLUMNS)
    con.execute(f"CREATE TABLE candle_expectancy_events ({ddl})")
    base = {name: None for name in names}
    rows = []
    for label, realized in [
        (TARGET_FIRST, 1.0),
        (STOP_FIRST, -1.0),
        (TARGET_FIRST, 1.0),
    ]:
        row = {
            **base,
            "symbol": "SPY",
            "session_date": "2026-01-02",
            "ts": "2026-01-02T10:00:00-05:00",
            "event_name": "bullish_conviction",
            "event_direction": "CALLS",
            "nearest_reference_name": "VWAP",
            "nearest_reference_relation": "above",
            "nearest_reference_distance_pct": 0.0005,
            "acceptance_state": "accepted_above",
            "volume_confirmation": "confirmed",
            "vol_state": "LOW_VOL",
            "macro_state": "RISK_ON",
            "dp_state": "NORMAL",
            "regime_label": "trend",
            "open_regime_label": "OPEN_DRIVE",
            "minutes_since_open": 45,
            "target_before_stop_label": label,
            "two_sided_first_touch": "up_target_first",
            "realized_R": realized,
            "favorable_excursion_pct": 0.002,
            "adverse_excursion_pct": 0.001,
            "forward_bar_count": 4,
            "horizon_bars": 4,
        }
        rows.append(row)
    con.executemany(
        f"INSERT INTO candle_expectancy_events ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
        [[row.get(name) for name in names] for row in rows],
    )

    build_matrix(
        con, "candle_expectancy_events", "candle_conditioned_expectancy_matrix"
    )
    out = con.execute("SELECT * FROM candle_conditioned_expectancy_matrix").fetchall()
    cur = con.execute(
        "SELECT * FROM candle_conditioned_expectancy_matrix WHERE match_tier='tier_1_full'"
    )
    cols = [desc[0] for desc in cur.description]
    packet = dict(zip(cols, cur.fetchone()))

    assert len(out) == 4
    assert packet["match_tier"] == "tier_1_full"
    assert packet["n"] == 3
    assert round(packet["target_before_stop_rate"], 3) == 0.667
    assert round(packet["avg_realized_R"], 3) == 0.333
    assert packet["reference_distance_bucket"] == "at_reference"
    assert packet["time_bucket"] == "opening_60m"


def test_candle_confidence_caps_lucky_small_samples_and_flags_supported_rows():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "match_tier": "tier_1_full",
                "event_name": "bullish_conviction",
                "event_direction": "CALLS",
                "nearest_reference_name": "VWAP",
                "nearest_reference_relation": "above",
                "reference_distance_bucket": "at_reference",
                "acceptance_state": "accepted_above",
                "volume_confirmation": "confirmed",
                "vol_state": "LOW_VOL",
                "macro_state": "RISK_ON",
                "dp_state": "NORMAL",
                "regime_label": "trend",
                "open_regime_label": "OPEN_DRIVE",
                "time_bucket": "opening_60m",
                "n": 3,
                "target_before_stop_rate": 1.0,
                "stop_before_target_rate": 0.0,
                "same_bar_rate": 0.0,
                "no_resolution_rate": 0.0,
                "up_target_first_rate": 0.9,
                "down_target_first_rate": 0.1,
                "avg_realized_R": 1.0,
                "avg_favorable_excursion_pct": 0.003,
                "avg_adverse_excursion_pct": 0.0005,
            },
            {
                "match_tier": "tier_2_execution",
                "event_name": "bearish_conviction",
                "event_direction": "PUTS",
                "nearest_reference_name": "PDH",
                "nearest_reference_relation": "below",
                "reference_distance_bucket": "near_reference",
                "acceptance_state": "accepted_below",
                "volume_confirmation": "confirmed",
                "vol_state": "HIGH_VOL",
                "macro_state": "RISK_OFF",
                "dp_state": "NORMAL",
                "regime_label": "trend",
                "open_regime_label": "FAILED_BREAK",
                "time_bucket": "midday",
                "n": 90,
                "target_before_stop_rate": 0.72,
                "stop_before_target_rate": 0.18,
                "same_bar_rate": 0.02,
                "no_resolution_rate": 0.08,
                "up_target_first_rate": 0.25,
                "down_target_first_rate": 0.65,
                "avg_realized_R": 0.54,
                "avg_favorable_excursion_pct": 0.0028,
                "avg_adverse_excursion_pct": 0.0008,
            },
        ]
    )

    confidence = build_confidence(df)
    small = confidence[confidence["n"] == 3].iloc[0]
    supported = confidence[confidence["n"] == 90].iloc[0]

    assert small["sample_bucket"] == "MICRO_SAMPLE"
    assert small["confidence_score"] <= 25
    assert small["deployment_tier"] == "RESEARCH_ONLY"
    assert supported["sample_bucket"] == "DEEP_SAMPLE"
    assert supported["deployment_ready"] == 1
    assert supported["deployment_tier"] in {
        "COCKPIT_SURFACE_ELIGIBLE",
        "WATCHLIST_OR_PROBE_ELIGIBLE",
    }
