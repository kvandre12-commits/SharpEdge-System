from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_risk_decision_layer import select_daily_records  # noqa: E402


def test_select_daily_records_keeps_one_best_row_per_symbol_date():
    records = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "date": "2026-06-10",
                "deployment_confidence": 0.40,
                "tradability_score": 0.90,
                "sample_n": 80,
                "deployment_state": "WATCH",
            },
            {
                "symbol": "SPY",
                "date": "2026-06-10",
                "deployment_confidence": 0.70,
                "tradability_score": 0.20,
                "sample_n": 5,
                "deployment_state": "PROBE",
            },
            {
                "symbol": "SPY",
                "date": "2026-06-10",
                "deployment_confidence": 0.70,
                "tradability_score": 0.80,
                "sample_n": 4,
                "deployment_state": "PROBE_BETTER_TRADEABILITY",
            },
            {
                "symbol": "SPY",
                "date": "2026-06-11",
                "deployment_confidence": 0.50,
                "tradability_score": 0.60,
                "sample_n": 10,
                "deployment_state": "PROBE_NEXT_DAY",
            },
        ]
    )

    selected = select_daily_records(records)

    assert len(selected) == 2
    assert selected[["symbol", "date"]].duplicated().sum() == 0
    first = selected[selected["date"].eq("2026-06-10")].iloc[0]
    assert first["deployment_state"] == "PROBE_BETTER_TRADEABILITY"


def test_select_daily_records_uses_sample_n_as_final_tiebreaker():
    records = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "date": "2026-06-10",
                "deployment_confidence": 0.70,
                "tradability_score": 0.80,
                "sample_n": 4,
                "deployment_state": "LOWER_SAMPLE",
            },
            {
                "symbol": "SPY",
                "date": "2026-06-10",
                "deployment_confidence": 0.70,
                "tradability_score": 0.80,
                "sample_n": 12,
                "deployment_state": "HIGHER_SAMPLE",
            },
        ]
    )

    selected = select_daily_records(records)

    assert len(selected) == 1
    assert selected.iloc[0]["deployment_state"] == "HIGHER_SAMPLE"
