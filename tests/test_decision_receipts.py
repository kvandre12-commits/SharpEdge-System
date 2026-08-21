from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from decision_receipts import (
    append_decision_receipt,
    build_decision_receipt,
    build_permission_score_trend,
    enrich_receipt_outcome,
    load_recent_receipts,
    update_receipt_outcomes,
)


def _permission(volume: int, acceptance: int, trend: int, balance: int) -> dict:
    return {
        "trade_gate": "CAUTION",
        "trade_permission_score": 66,
        "execution_permission_score": 66,
        "bias": "NEUTRAL",
        "setup_conviction": {
            "setup_conviction_score": 84,
            "setup_gate": "ACTIONABLE",
            "bias": "CALLS",
            "setup_tag": "FAILED BREAKDOWN",
            "reason": "reclaimed support",
        },
        "scores": {
            "volume_score": {"score": volume, "reason": "participation confirms move"},
            "acceptance_score": {"score": acceptance, "reason": "accepted above level"},
            "trend_score": {"score": trend, "reason": "trend weakening"},
            "expansion_fuel_score": {
                "score": 90,
                "bias": "CALLS",
                "reason": "expansion fuel is active",
            },
            "balance_context_score": {
                "score": balance,
                "reason": "balance disagreement",
            },
        },
    }


def test_build_decision_receipt_persists_reasoning_chain():
    receipt = build_decision_receipt(
        "2026-06-25T11:15:00",
        "SPY",
        733.24,
        _permission(85, 78, 58, 28),
        {
            "label": "Magnet",
            "price": 740.0,
            "status": "beyond",
            "distance": 6.76,
            "reachable_today": {"label": "VWAP", "price": 734.5},
            "likely_travel": "partial reversion only",
        },
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            }
        ],
    )

    assert receipt["permission"] == 66
    assert receipt["execution_permission"] == 66
    assert receipt["setup_conviction"]["setup_gate"] == "ACTIONABLE"
    assert receipt["setup"] == "STICKY DAY (calm/chop)"
    assert receipt["strategic_target"]["label"] == "Magnet"
    assert receipt["reachable_today"]["label"] == "VWAP"
    assert receipt["top_trade"] == ["Participation", "Auction Acceptance", "Trend"]
    assert receipt["top_wait"] == ["Balance Context", "Trend", "Auction Acceptance"]
    assert receipt["outcome"] is None


def test_build_decision_receipt_prefers_actionable_trade_gate_over_context_card():
    receipt = build_decision_receipt(
        "2026-06-25T10:05:00",
        "SPY",
        733.1,
        _permission(80, 76, 55, 30),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            },
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            },
        ],
    )

    assert receipt["setup"] == "FAILED BREAKDOWN"
    assert receipt["entry_gate"]["gate_id"] == "failed_breakdown_reclaim"
    assert receipt["entry_gate"]["level_name"] == "ORL"
    assert receipt["context_gate"]["gate_id"] == "sticky_day_magnet_fade"


def test_build_permission_score_trend_tracks_score_points_and_feature_deltas():
    previous = build_decision_receipt(
        "2026-06-25T09:45:00",
        "SPY",
        733.0,
        _permission(73, 70, 68, 35),
        {"label": "VWAP", "price": 734.5},
        [{"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)"}],
    )
    current = build_decision_receipt(
        "2026-06-25T11:15:00",
        "SPY",
        733.24,
        _permission(85, 78, 58, 28),
        {"label": "Magnet", "price": 740.0},
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            }
        ],
    )

    trend = build_permission_score_trend(current, [previous])

    assert trend["direction"] == "flat"
    assert trend["current"] == 66
    assert trend["previous"] == 66
    assert [point["time"] for point in trend["points"]] == ["09:45", "11:15"]
    assert trend["points"][0]["event_markers"] == ["FAILED BREAKDOWN CANDIDATE"]
    assert trend["points"][1]["event_markers"] == ["STICKY DAY (calm/chop) CANDIDATE"]
    deltas = {
        item["feature"]: item["delta"]
        for item in trend["largest_changes_since_last_update"]
    }
    assert deltas["Participation"] == 12
    assert deltas["Auction Acceptance"] == 8
    assert deltas["Trend"] == -10
    assert deltas["Balance Context"] == -7


def test_setup_events_get_lifecycle_candidate_confirmed_then_expired():
    candidate = build_decision_receipt(
        "2026-06-25T10:12:00",
        "SPY",
        733.0,
        _permission(72, 74, 60, 32),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
    )
    confirmed = build_decision_receipt(
        "2026-06-25T10:15:00",
        "SPY",
        733.3,
        _permission(78, 80, 64, 30),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
        previous_receipt=candidate,
    )
    expired = build_decision_receipt(
        "2026-06-25T10:41:00",
        "SPY",
        733.8,
        _permission(70, 62, 54, 42),
        {"label": "Magnet", "price": 734.0},
        [],
        previous_receipt=confirmed,
    )

    candidate_event = candidate["setup_events"][0]
    confirmed_event = confirmed["setup_events"][0]
    expired_event = expired["setup_events"][0]

    assert candidate_event["status"] == "candidate"
    assert candidate["setup_event_transitions"][0]["transition"] == "new"
    assert confirmed_event["status"] == "confirmed"
    assert confirmed_event["first_seen_ts"] == "2026-06-25T10:12:00"
    assert confirmed_event["last_confirmed_ts"] == "2026-06-25T10:15:00"
    assert confirmed["setup_event_transitions"][0]["transition"] == "promoted"
    assert expired_event["status"] == "expired"
    assert expired["setup_event_transitions"][0]["transition"] == "expired"


def test_confirmed_actionable_setup_persists_through_context_only_tick():
    confirmed = build_decision_receipt(
        "2026-06-25T10:15:00",
        "SPY",
        733.3,
        _permission(78, 80, 64, 30),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "detail": "reclaimed ORL",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
        previous_receipt=build_decision_receipt(
            "2026-06-25T10:12:00",
            "SPY",
            733.0,
            _permission(72, 74, 60, 32),
            {"label": "VWAP", "price": 734.5},
            [
                {
                    "tag": "FAILED BREAKDOWN",
                    "bias": "CALLS (bullish)",
                    "kind": "ok",
                    "detail": "reclaimed ORL",
                    "level_name": "ORL",
                    "level_price": 732.9,
                    "trigger_price": 732.4,
                    "bars_ago": 1,
                }
            ],
        ),
    )
    sticky_only = build_decision_receipt(
        "2026-06-25T10:18:00",
        "SPY",
        733.4,
        _permission(70, 62, 54, 42),
        {"label": "Magnet", "price": 734.0},
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
                "kind": "info",
                "detail": "positive gamma context only",
            }
        ],
        previous_receipt=confirmed,
    )

    assert sticky_only["setup"] == "FAILED BREAKDOWN"
    assert sticky_only["entry_gate"]["gate_id"] == "failed_breakdown_reclaim"
    assert sticky_only["context_gate"]["gate_id"] == "sticky_day_magnet_fade"
    assert sticky_only["primary_setup_event"]["status"] == "confirmed"
    assert sticky_only["primary_setup_event"]["persisted_without_fresh_trigger"] is True


def test_append_and_load_recent_receipts_round_trip(tmp_path):
    receipt_path = tmp_path / "permission_receipts_spy.jsonl"
    first = build_decision_receipt(
        "2026-06-25T08:30:00",
        "SPY",
        732.5,
        _permission(60, 54, 49, 40),
        {"label": "VWAP", "price": 734.5},
        [],
    )
    second = build_decision_receipt(
        "2026-06-25T09:45:00",
        "SPY",
        733.0,
        _permission(73, 70, 68, 35),
        {"label": "Magnet", "price": 740.0},
        [],
    )

    append_decision_receipt(receipt_path, first)
    append_decision_receipt(receipt_path, second)

    rows = load_recent_receipts(receipt_path)

    assert len(rows) == 2
    assert rows[0]["ts"] == "2026-06-25T08:30:00"
    assert rows[1]["strategic_target"]["label"] == "Magnet"
    assert json.loads(receipt_path.read_text().splitlines()[0])["symbol"] == "SPY"


def test_enrich_receipt_outcome_uses_truth_and_trade_rows():
    receipt = build_decision_receipt(
        "2026-06-25T11:15:00",
        "SPY",
        733.24,
        _permission(85, 78, 58, 28),
        {
            "label": "Magnet",
            "price": 740.0,
            "reachable_today": {"label": "Channel hi", "price": 733.35},
        },
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            }
        ],
    )

    outcome = enrich_receipt_outcome(
        receipt,
        {"high": 734.0, "low": 730.0, "close": 731.5},
        {"trade_taken": True, "trade_count": 1, "pnl_r": 0.8, "realized_pnl": 54.0},
    )

    assert outcome["target_reached"] is False
    assert outcome["reachable_today_reached"] is True
    assert outcome["max_excursion"] == 0.76
    assert outcome["trade_taken"] is True
    assert outcome["pnl_r"] == 0.8


def test_update_receipt_outcomes_rewrites_jsonl_and_signal(tmp_path):
    receipt_path = tmp_path / "outputs/permission_receipts_spy.jsonl"
    truth_path = tmp_path / "outputs/spy_truth_daily.csv"
    signal_path = tmp_path / "outputs/signal.json"
    db_path = tmp_path / "data/spy_truth.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    receipt = build_decision_receipt(
        "2026-06-25T11:15:00",
        "SPY",
        733.24,
        _permission(85, 78, 58, 28),
        {
            "label": "Magnet",
            "price": 740.0,
            "reachable_today": {"label": "Channel hi", "price": 733.35},
        },
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            }
        ],
    )
    append_decision_receipt(receipt_path, receipt)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        "date,symbol,open,high,low,close\n2026-06-25,SPY,733.0,734.0,730.0,731.5\n",
        encoding="utf-8",
    )
    signal_path.write_text(
        json.dumps({"decision_receipt": receipt}, indent=2), encoding="utf-8"
    )

    summary = update_receipt_outcomes(receipt_path, truth_path, db_path, signal_path)
    rows = load_recent_receipts(receipt_path)
    signal = json.loads(signal_path.read_text(encoding="utf-8"))

    assert summary["updated_count"] == 1
    assert summary["signal_updated"] is True
    assert rows[0]["outcome"]["target_reached"] is False
    assert rows[0]["outcome"]["reachable_today_reached"] is True
    assert signal["decision_receipt"]["outcome"]["max_excursion"] == 0.76
