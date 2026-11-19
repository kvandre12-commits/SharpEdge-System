from __future__ import annotations

from datetime import UTC, datetime

from scripts.agents.helios_call_ladder_runner import (
    _due,
    _features,
    _symbol_cycle_state,
)


def test_due_only_inside_weekday_1230_window():
    assert _due(datetime(2026, 11, 18, 17, 30, tzinfo=UTC)) is True
    assert _due(datetime(2026, 11, 18, 17, 40, tzinfo=UTC)) is True
    assert _due(datetime(2026, 11, 18, 17, 41, tzinfo=UTC)) is False
    assert _due(datetime(2026, 11, 21, 17, 30, tzinfo=UTC)) is False


def test_features_compute_vwap_momentum_and_volume_ratio():
    rows = [
        {
            "open": 100 + index * 0.1,
            "high": 100.2 + index * 0.1,
            "low": 99.8 + index * 0.1,
            "close": 100.1 + index * 0.1,
            "volume": 100 + index * 10,
        }
        for index in range(20)
    ]

    features = _features(rows)

    assert features is not None
    assert features["spot"] == 102.0
    assert features["vs_vwap_pct"] > 0
    assert features["momentum_15m_pct"] > 0
    assert features["volume_ratio"] > 1


def test_old_unfilled_proposal_resets_next_session_but_open_position_does_not():
    state = {
        "symbols": {
            "SPY": {"state": "ENTRY_PROPOSED", "session_date": "2026-11-17"},
            "NVDA": {"state": "OPEN_POSITION", "session_date": "2026-11-17"},
        }
    }

    assert _symbol_cycle_state(state, "SPY", "2026-11-18") == "IDLE"
    assert _symbol_cycle_state(state, "NVDA", "2026-11-18") == "OPEN_POSITION"
