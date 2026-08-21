from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.agents.operator_decision_card import build_card, render_html

NOW = datetime(2026, 8, 20, 15, 0, tzinfo=UTC)


def _signal(**updates):
    signal = {
        "ts": NOW.isoformat(),
        "symbol": "SPY",
        "spot": 101.0,
        "vwap": 100.0,
        "ema9": 100.5,
        "ema20": 99.5,
        "ema9_slope_5": 0.2,
        "entry_setup_tag": "DOWNSIDE EXHAUSTION",
        "entry_setup_bias": "watch for reversal UP (calls)",
        "gamma_regime": "negative",
        "vol_mult": 1.8,
        "mom15": 0.2,
    }
    signal.update(updates)
    return signal


def _expression(direction="CALLS", family="call_debit_spread"):
    return {
        "branch_expressions": [
            {
                "direction": direction,
                "structure_family": family,
                "structure_label": "101/103 call debit spread",
                "expression_objective": "defined-risk upside participation",
                "greek_dollar_plan": {"defined_risk": {"max_loss_dollars": 75}},
            }
        ]
    }


def test_blocked_card_still_shows_conditional_calls_and_bounce_map():
    card = build_card(
        _signal(),
        {},
        {"trade_allowed": False, "blocking_reasons": ["stale_or_missing_inputs"]},
        now=NOW,
    )

    assert card["state"] == "BLOCKED"
    assert card["direction"] == "CALLS"
    assert card["levels"]["confirmation_level"] == 100.5
    assert card["levels"]["invalidation_level"] == 99.5
    assert card["levels"]["bounce_zone"]["members"] == ["EMA9", "VWAP"]
    assert card["option_expression"]["family"] == "unavailable"
    assert "approval_trade_not_allowed" in card["authority"]["blockers"]


def test_approved_confirmed_calls_selects_quote_validated_debit_spread():
    card = build_card(
        _signal(),
        _expression(),
        {"trade_allowed": True, "broker_order_allowed": False, "decision": "permit"},
        now=NOW,
    )

    assert card["state"] == "TRIGGER_READY"
    assert card["action"] == "TRIGGER_READY — CALLS via 101/103 call debit spread"
    assert card["option_expression"]["family"] == "call_debit_spread"
    assert card["authority"]["operator_confirmation_required"] is True


def test_upside_exhaustion_maps_to_puts_and_bearish_confirmation():
    card = build_card(
        _signal(
            spot=99.0,
            ema9=99.5,
            ema20=100.5,
            entry_setup_tag="UPSIDE EXHAUSTION",
            entry_setup_bias="watch for reversal DOWN (puts)",
        ),
        _expression(direction="PUTS", family="long_put"),
        {"trade_allowed": True},
        now=NOW,
    )

    assert card["direction"] == "PUTS"
    assert card["levels"]["confirmation_level"] == 99.5
    assert card["levels"]["invalidation_level"] == 100.5
    assert card["option_expression"]["family"] == "long_put"


def test_stale_signal_blocks_otherwise_ready_card():
    card = build_card(
        _signal(ts=(NOW - timedelta(minutes=3)).isoformat()),
        _expression(),
        {"trade_allowed": True},
        now=NOW,
    )

    assert card["state"] == "BLOCKED"
    assert "signal_stale" in card["authority"]["blockers"]


def test_html_is_one_screen_semantic_artifact():
    card = build_card(_signal(), _expression(), {"trade_allowed": True}, now=NOW)

    page = render_html(card)

    assert page.count("<main>") == 1
    assert "Bounce / rejection zone" in page
    assert "Option expression" in page
    assert "Confirmation" in page
    assert "Invalidation" in page
