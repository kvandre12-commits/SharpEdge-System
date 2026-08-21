from __future__ import annotations

from datetime import date

from scripts.agents.position_steward_logic import (
    build_position_snapshot,
    build_recovery_snapshot,
    build_session_snapshot,
    classify_action,
)


def _spec(*, status="active", risk_budget=150.0):
    return {
        "symbol": "GME",
        "thesis": {
            "anchor_low": 18.55,
            "anchor_high": 21.72,
            "invalidation_price": 18.55,
        },
        "risk": {
            "max_total_debit_dollars": risk_budget,
            "target_delta_shares": 35,
        },
        "management": {
            "roll_short_dte": 5,
            "grow_min_recovery_pct": 25,
            "grow_max_recovery_pct": 61.8,
            "trim_recovery_pct": 75,
        },
        "position": {"status": status},
    }


def _position(**overrides):
    base = {
        "available": True,
        "entry_debit_dollars": 94.0,
        "estimated_pnl_dollars": 10.0,
        "net_delta_shares": 20.0,
        "nearest_short_dte": 20,
    }
    return {**base, **overrides}


def _session(**overrides):
    return {"available": True, "above_vwap": True, "momentum_15m_pct": 0.2, **overrides}


def _fresh(**overrides):
    return {"market_open": True, "fresh_for_management": True, **overrides}


def test_recovery_snapshot_builds_percentage_ladder():
    snapshot = build_recovery_snapshot(19.16, 18.55, 21.72)
    assert snapshot["recovery_pct"] == 19.2
    assert snapshot["phase"] == "base_building"
    assert snapshot["ladder"]["50%"] == 20.14
    assert snapshot["ladder"]["61.8%"] == 20.51


def test_unconfirmed_or_null_budget_fails_closed():
    recovery = build_recovery_snapshot(19.6, 18.55, 21.72)
    for spec in (_spec(status="research"), _spec(risk_budget=None)):
        action = classify_action(
            spec,
            spot=19.6,
            recovery=recovery,
            session=_session(),
            position=_position(),
            freshness=_fresh(),
        )
        assert action["state"] == "insufficient_position_context"
        assert "add" not in action["reason"].lower()


def test_market_closed_and_stale_evidence_do_not_create_actionable_advice():
    recovery = build_recovery_snapshot(19.6, 18.55, 21.72)
    closed = classify_action(
        _spec(),
        spot=19.6,
        recovery=recovery,
        session=_session(),
        position=_position(),
        freshness=_fresh(market_open=False),
    )
    stale = classify_action(
        _spec(),
        spot=19.6,
        recovery=recovery,
        session=_session(),
        position=_position(),
        freshness=_fresh(fresh_for_management=False),
    )
    assert closed["state"] == "market_closed_review"
    assert stale["state"] == "refresh_required"


def test_invalidation_and_short_expiry_take_precedence():
    broken = classify_action(
        _spec(),
        spot=18.5,
        recovery=build_recovery_snapshot(18.5, 18.55, 21.72),
        session=_session(),
        position=_position(nearest_short_dte=2),
        freshness=_fresh(),
    )
    roll = classify_action(
        _spec(),
        spot=19.6,
        recovery=build_recovery_snapshot(19.6, 18.55, 21.72),
        session=_session(),
        position=_position(nearest_short_dte=2),
        freshness=_fresh(),
    )
    assert broken["state"] == "exit_thesis_broken"
    assert roll["state"] == "roll_short_leg"


def test_consider_add_requires_every_growth_gate():
    recovery = build_recovery_snapshot(19.6, 18.55, 21.72)
    action = classify_action(
        _spec(),
        spot=19.6,
        recovery=recovery,
        session=_session(),
        position=_position(),
        freshness=_fresh(),
    )
    loser = classify_action(
        _spec(),
        spot=19.6,
        recovery=recovery,
        session=_session(),
        position=_position(estimated_pnl_dollars=-1.0),
        freshness=_fresh(),
    )
    assert action["state"] == "consider_add"
    assert action["operator_approval_required"] is True
    assert loser["state"] == "hold"


def test_position_snapshot_uses_conservative_liquidation_and_signed_greeks():
    position = {
        "entry_debit_dollars": 94.0,
        "legs": [
            {
                "contract": "LONG",
                "side": "buy",
                "quantity": 1,
                "expiration": "2026-10-16",
                "strike": 20,
            },
            {
                "contract": "SHORT",
                "side": "sell",
                "quantity": 1,
                "expiration": "2026-09-18",
                "strike": 20,
            },
        ],
    }
    quotes = {
        "LONG": {
            "bid": 1.0,
            "ask": 1.1,
            "delta": 0.45,
            "gamma": 0.12,
            "theta": -0.01,
            "vega": 0.03,
        },
        "SHORT": {
            "bid": 0.6,
            "ask": 0.7,
            "delta": 0.30,
            "gamma": 0.15,
            "theta": -0.02,
            "vega": 0.02,
        },
    }
    snapshot = build_position_snapshot(position, quotes, as_of=date(2026, 8, 10))
    assert snapshot["available"] is True
    assert snapshot["conservative_liquidation_value_dollars"] == 30.0
    assert snapshot["estimated_pnl_dollars"] == -64.0
    assert snapshot["net_delta_shares"] == 15.0
    assert snapshot["theta_dollars_per_day"] == 1.0
    assert snapshot["nearest_short_dte"] == 39


def test_session_snapshot_computes_vwap_and_momentum():
    rows = [
        {
            "date": "2026-08-10",
            "open": 19.0,
            "high": 19.1,
            "low": 18.9,
            "close": 19.0,
            "volume": 100,
        },
        {
            "date": "2026-08-10",
            "open": 19.0,
            "high": 19.2,
            "low": 19.0,
            "close": 19.1,
            "volume": 100,
        },
        {
            "date": "2026-08-10",
            "open": 19.1,
            "high": 19.3,
            "low": 19.1,
            "close": 19.2,
            "volume": 100,
        },
        {
            "date": "2026-08-10",
            "open": 19.2,
            "high": 19.4,
            "low": 19.2,
            "close": 19.3,
            "volume": 100,
        },
    ]
    session = build_session_snapshot(rows)
    assert session["available"] is True
    assert session["above_vwap"] is True
    assert session["momentum_15m_pct"] > 0
