from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES  # noqa: E402
from trade_permission import (  # noqa: E402
    AUTHORITY_ENGINE_ENV_VAR,
    resolve_authority_engine,
    score_trade_permission,
)


def _bull_bars():
    bars = []
    price = 100.0
    for minute in range(45):
        open_ = price
        close = price + 0.08
        high = close + 0.04
        low = open_ - 0.02
        volume = 1000 + minute * 12
        bars.append((minute, open_, high, low, close, volume))
        price = close
    return bars


def _pa(bars, **overrides):
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "balance_high": closes[-2],
        "balance_low": closes[-8],
        "position_in_balance": 1.0,
        "balance_state": "above",
        "balance_label": "TOP",
        "balance_width_pct": 0.42,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "recent_balance",
        "dominant_balance_reason": "mid-session: the active recent box matters most",
        "session_position_in_range": 0.92,
        "rng_pos": 92.0,
        "day_chg": 1.2,
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
    }
    pa.update(overrides)
    return pa


def test_resolve_authority_engine_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv(AUTHORITY_ENGINE_ENV_VAR, raising=False)
    assert resolve_authority_engine() == "legacy"


def test_score_trade_permission_can_switch_to_ace_authority_with_env(monkeypatch):
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}

    monkeypatch.setenv(AUTHORITY_ENGINE_ENV_VAR, "ace")
    card = score_trade_permission(
        bars, pa, levels, [], op, {}, {"premium_read": "cheap"}
    )

    assert card["authority_engine"] == "ace"
    assert card["authority_mode"] == "core_spine_only"
    assert card["trade_permission_score"] == card["bucket_conditioned_spine"]["score"]
    assert set(card["scores"]) == set(CORE_EXECUTION_SPINE_PART_NAMES)
    assert card["execution_hierarchy"]["secondary_confirmations"] == []
    assert card["execution_hierarchy"]["context_governors"] == []
    assert card["execution_hierarchy"]["suspect_drift_voices"] == []
    assert card["market_day"]["bucket"]
    assert card["authority_summary"]["bucket"] == card["market_day"]["bucket"]
    assert card["authority_summary"]["diagnostic_posture"]
    audit = card["authority_self_audit"]
    assert audit["status"] == "demoted_pending_calibration"
    assert audit["score_spine_role"] == "diagnostic_advisory"
    assert audit["final_authority_source"] == "approval_decision_plus_operator"
    assert audit["tightened_facts"]


def test_explicit_authority_engine_argument_overrides_environment(monkeypatch):
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    monkeypatch.setenv(AUTHORITY_ENGINE_ENV_VAR, "ace")
    legacy_card = score_trade_permission(
        bars,
        pa,
        levels,
        [],
        {"atm_iv": 0.18},
        {},
        {},
        authority_engine="legacy",
    )

    assert legacy_card.get("authority_engine") == "legacy"
    assert legacy_card.get("authority_mode") == "full_stack"
    assert (
        legacy_card["authority_adjudication"]["cockpit_read"]["authority_engine"]
        == "legacy"
    )
    assert (
        legacy_card["authority_adjudication"]["we_are_doing_this"]["authority_engine"]
        == "legacy"
    )
    ace_voice = next(
        voice
        for voice in legacy_card["authority_adjudication"]["competing_voices"]
        if voice.get("voice_id") == "ace_advisory"
    )
    assert ace_voice["advisory_only"] is True
    assert ace_voice["engine"] == "ace"
    assert (
        legacy_card["authority_adjudication"]["advisory_engines"][0]["engine"] == "ace"
    )
    assert "pressure_score" in legacy_card["scores"]
    assert legacy_card["execution_hierarchy"]["suspect_drift_voices"]


def test_ace_can_keep_failed_break_bucket_context_while_excluding_trap_and_rejection(
    monkeypatch,
):
    bars = [
        (0, 100.28, 100.32, 100.10, 100.20, 950),
        (1, 100.20, 100.24, 100.02, 100.12, 980),
        (2, 100.20, 100.24, 100.00, 100.08, 1000),
        (3, 100.08, 100.12, 99.96, 100.00, 1100),
        (4, 100.00, 100.04, 99.72, 99.84, 1200),
        (5, 99.84, 99.90, 99.60, 99.78, 1300),
        (6, 99.78, 100.00, 99.74, 99.94, 1400),
        (7, 99.94, 100.08, 99.90, 100.02, 1500),
    ]
    pa = _pa(
        bars,
        spot=100.02,
        hi=100.08,
        lo=99.78,
        rng_pos=40.0,
        vs_vwap=0.06,
        mom15=0.08,
        vol_mult=1.1,
    )
    levels = {"ORH": 100.40, "ORL": 99.80, "PDH": 100.55, "PDL": 99.20}
    setup = {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": "reclaimed ORL after flush",
        "level_name": "ORL",
        "level_price": 99.80,
        "trigger_price": 99.60,
        "bars_ago": 1,
    }

    monkeypatch.setenv(AUTHORITY_ENGINE_ENV_VAR, "ace")
    card = score_trade_permission(
        bars,
        pa,
        levels,
        [setup],
        {"atm_iv": 0.18, "call_wall": 101.0, "put_wall": 99.0},
        {"regime": "positive", "pin": 100.0},
        {},
    )

    assert card["authority_engine"] == "ace"
    assert card["market_day"]["bucket"] == "failed_breakdown_long_day"
    assert card["authority_summary"]["bucket"] == "failed_breakdown_long_day"
    assert card["setup_conviction"]["setup_tag"] == "FAILED BREAKDOWN"
    assert "trap_score" not in card["scores"]
    assert "rejection_score" not in card["scores"]
    assert card["live_trap_corroboration"]["trap_score"] >= 78
    assert card["live_trap_corroboration"]["trap_bias"] == "CALLS"
    assert card["execution_hierarchy"]["secondary_confirmations"] == []
