from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from acceptance_state_engine import build_acceptance_state  # noqa: E402
import execution_vector_context as ctx  # noqa: E402
from trade_permission import ExecutionVectorEngine  # noqa: E402


def _accepted_above_bars() -> list[tuple]:
    closes = [99.92, 100.05, 100.18, 100.25, 100.32, 100.40]
    bars = []
    for minute, close in enumerate(closes):
        open_ = closes[minute - 1] if minute else close - 0.05
        high = max(open_, close) + 0.04
        low = min(open_, close) - 0.04
        bars.append((minute, open_, high, low, close, 1000 + minute * 20))
    return bars


def _no_acceptance_bars() -> list[tuple]:
    closes = [100.12, 100.18, 100.15, 100.22, 100.18]
    bars = []
    for minute, close in enumerate(closes):
        open_ = closes[minute - 1] if minute else close - 0.03
        high = max(open_, close) + 0.03
        low = min(open_, close) - 0.03
        bars.append((minute, open_, high, low, close, 900 + minute * 10))
    return bars


def test_acceptance_state_exposes_all_accepted_levels():
    packet = build_acceptance_state(
        _accepted_above_bars(),
        {"PDH": 99.90, "ORH": 99.95, "PDC": 100.00},
    )

    assert packet["state"] == "accepted_above_level"
    assert packet["bias"] == "CALLS"
    assert packet["accepted_level_count"] == 3
    assert packet["representative_level"]["level_name"] == "PDC"
    assert {item["level_name"] for item in packet["accepted_levels"]} == {
        "PDH",
        "ORH",
        "PDC",
    }


def test_acceptance_state_stays_neutral_without_clean_level_acceptance():
    packet = build_acceptance_state(
        _no_acceptance_bars(),
        {"ORL": 100.10, "PDC": 100.20, "ORH": 100.30, "VWAP": 100.13},
    )

    assert packet["state"] == "no_acceptance"
    assert packet["bias"] == "NEUTRAL"
    assert packet["evaluated_levels"] == ["ORH", "ORL", "PDC", "VWAP"]
    assert packet["accepted_level_count"] == 0
    assert packet["reason"] == "no clean level acceptance"


def test_acceptance_level_map_filters_to_standardized_candidate_levels():
    levels = ctx.acceptance_level_map(
        {"vwap": 100.13},
        {"ORL": 100.10, "PDC": 100.20, "ORH": 100.30, "ONH": 100.50},
        {"call_wall": 100.7, "put_wall": 99.4},
    )

    assert levels == {"ORH": 100.30, "ORL": 100.10, "PDC": 100.20}


def test_execution_vector_engine_acceptance_score_uses_acceptance_state_engine():
    engine = ExecutionVectorEngine()

    accepted_parts = engine.build_parts(
        _accepted_above_bars(),
        {"spot": 100.4, "vwap": 100.0, "vs_vwap": 0.2, "mom15": 0.3},
        {"PDH": 99.9, "ORH": 99.95, "PDC": 100.0, "ONH": 100.15},
        [],
        {"call_wall": 100.1, "put_wall": 99.2},
        {},
        {},
    )
    accepted = accepted_parts["acceptance_score"]
    accepted_state = dict(engine.acceptance_state)
    no_acceptance = engine.build_parts(
        _no_acceptance_bars(),
        {"spot": 100.18, "vwap": 100.13, "vs_vwap": 0.2, "mom15": 0.1},
        {"ORL": 100.10, "PDC": 100.20, "ORH": 100.30},
        [],
        {},
        {},
        {},
    )["acceptance_score"]

    assert accepted.score == 78
    assert accepted.reason == "3 closes accepted above PDC 100.00"
    assert accepted_state["evaluated_levels"] == ["ORH", "PDC", "PDH"]
    assert {item["level_name"] for item in accepted_state["accepted_levels"]} == {
        "ORH",
        "PDC",
        "PDH",
    }
    assert no_acceptance.score == 35
    assert no_acceptance.reason == "no clean level acceptance"
