from __future__ import annotations

from datetime import UTC, datetime

from scripts.agents import helios_call_ladder_runner as runner
from scripts.agents.helios_call_ladder_runner import (
    _apply_global_wallet,
    _build_disagreement_receipts,
    _due,
    _features,
    _wallet_cycle_state,
)


def test_due_only_inside_weekday_noon_to_1230_window():
    assert _due(datetime(2026, 11, 18, 17, 0, tzinfo=UTC)) is True
    assert _due(datetime(2026, 11, 18, 17, 30, tzinfo=UTC)) is True
    assert _due(datetime(2026, 11, 18, 17, 31, tzinfo=UTC)) is False
    assert _due(datetime(2026, 11, 21, 17, 0, tzinfo=UTC)) is False


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
    old_proposal = {
        "symbols": {
            "SPY": {"state": "ENTRY_PROPOSED", "session_date": "2026-11-17"},
        }
    }
    open_position = {
        "symbols": {
            "NVDA": {"state": "OPEN_POSITION", "session_date": "2026-11-17"},
        }
    }

    assert _wallet_cycle_state(old_proposal, "2026-11-18") == "IDLE"
    assert _wallet_cycle_state(open_position, "2026-11-18") == "OPEN_POSITION"


def test_missing_bridge_source_writes_blocked_handoff(tmp_path, monkeypatch):
    output = tmp_path / "wallet_handoff.json"
    monkeypatch.setattr(runner, "BRIDGE_ROOT", tmp_path / "missing-bridge")
    monkeypatch.setattr(runner, "WALLET_HANDOFF_PATH", output)

    runner._publish_wallet_handoff(tmp_path / "lead.json")

    payload = runner._read_json(output)
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["robinhood_bridge_source_unavailable"]


def test_prefill_disagreement_receipt_is_attached_to_lead_identity():
    lead = {
        "symbol": "SPY",
        "lead_id": "lead-1",
        "aggregate_decision": "approve",
        "source_hashes": {"nerv": "a" * 64},
        "reviews": [
            {"role": "data_steward", "decision": "approve", "reasons": []},
            {"role": "trigger_researcher", "decision": "approve", "reasons": []},
        ],
    }

    receipt = _build_disagreement_receipts([lead])[0]

    assert receipt["symbol"] == "SPY"
    assert receipt["lead_id"] == "lead-1"
    assert receipt["receipt"]["resolution"] == "ENTRY_PROPOSAL_ELIGIBLE"
    assert receipt["receipt"]["authority"]["execution_permitted"] is False


def test_global_wallet_never_allocates_more_than_250_across_symbols():
    def lead(symbol, costs):
        return {
            "symbol": symbol,
            "aggregate_decision": "approve",
            "ladder": [
                {
                    "contract_symbol": f"{symbol}{index}",
                    "max_debit_dollars": cost,
                    "score": 100 - index,
                }
                for index, cost in enumerate(costs)
            ],
            "wallet": {},
            "entry_handoff": {"state": "ENTRY_PROPOSED"},
            "reviews": [],
            "lead_id": "old",
        }

    leads = _apply_global_wallet([lead("SPY", [100, 90]), lead("NVDA", [80, 70])])

    total = sum(item["max_debit_dollars"] for lead in leads for item in lead["ladder"])
    assert total <= 250
    assert all(lead["wallet"]["global_proposed_debit_dollars"] == total for lead in leads)
