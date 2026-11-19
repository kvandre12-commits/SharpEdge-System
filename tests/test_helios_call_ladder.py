from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.alpha_swarm.helios_call_ladder import build_lead

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 18, 12, 30, tzinfo=NY)


def _features(**overrides):
    return {
        "vs_vwap_pct": 0.12,
        "momentum_15m_pct": 0.09,
        "volume_ratio": 1.5,
        **overrides,
    }


def _quote(contract, expiration, strike, ask, *, volume=500, oi=1000, bid=None):
    return {
        "underlying": "SPY",
        "contract_symbol": contract,
        "option_type": "call",
        "expiration": expiration,
        "strike": strike,
        "underlying_price": 730.0,
        "bid": ask - 0.05 if bid is None else bid,
        "ask": ask,
        "volume": volume,
        "open_interest": oi,
        "implied_volatility": 0.25,
        "delta": 0.30,
    }


def _build(**overrides):
    params = {
        "symbol": "SPY",
        "now": NOW,
        "features": _features(),
        "quotes": [
            _quote("SPY1", "2026-11-19", 731, 0.95),
            _quote("SPY2", "2026-11-20", 732, 0.80),
            _quote("SPY3", "2026-11-20", 734, 0.70),
        ],
        "nearest_target": 739.95,
        "source_hashes": {"nerv": "a" * 64, "features": "b" * 64},
    }
    params.update(overrides)
    return build_lead(**params)


def test_approved_ladder_stays_under_wallet_and_is_non_executable():
    payload = _build()

    assert payload["aggregate_decision"] == "approve"
    assert payload["entry_handoff"]["state"] == "ENTRY_PROPOSED"
    assert payload["wallet"]["proposed_debit_dollars"] <= 250
    assert len(payload["ladder"]) == 3
    assert {item["dte"] for item in payload["ladder"]} == {1, 2}
    assert all(item["strike"] > 730 for item in payload["ladder"])
    assert payload["authority"]["execution_permitted"] is False
    assert payload["authority"]["broker_action_allowed"] is False
    assert payload["entry_handoff"]["operator_confirmation_required"] is True
    thesis = payload["entry_handoff"]["entry_thesis_snapshot"]
    assert thesis["symbol"] == "SPY"
    assert thesis["owner_usage"]["may_tighten_exit"] is True
    assert thesis["owner_usage"]["may_loosen_risk_or_expiry_exit"] is False
    assert thesis["owner_usage"]["may_authorize_add_or_rebuy"] is False


def test_non_idle_cycle_rejects_rebuy_even_when_trigger_stays_true():
    payload = _build(cycle_state="OPEN_POSITION")

    assert payload["aggregate_decision"] == "reject"
    assert "cycle_not_idle:OPEN_POSITION" in payload["reviews"][-1]["reasons"]
    assert payload["entry_handoff"]["same_trigger_reentry_allowed"] is False


def test_failed_bullish_trigger_rejects_lead():
    payload = _build(features=_features(momentum_15m_pct=-0.1))

    assert payload["aggregate_decision"] == "reject"
    assert payload["trigger"]["triggered"] is False
    assert "momentum_15m_below_0.05pct" in payload["trigger"]["reasons"]


def test_stale_data_abstains_instead_of_approving():
    payload = _build(data_issues=["stale_nerv_snapshot"])

    assert payload["aggregate_decision"] == "abstain"
    assert "stale_nerv_snapshot" in payload["reviews"][0]["reasons"]


def test_missing_quotes_abstains_instead_of_inventing_contracts():
    payload = _build(quotes=[])

    assert payload["aggregate_decision"] == "abstain"
    assert payload["ladder"] == []


def test_liquidity_price_dte_and_option_type_are_hard_filters():
    quotes = [
        _quote("TOO_EXPENSIVE", "2026-11-19", 731, 1.01),
        _quote("TOO_WIDE", "2026-11-19", 732, 0.90, bid=0.40),
        _quote("NO_VOLUME", "2026-11-20", 733, 0.50, volume=0),
        _quote("NO_OI", "2026-11-20", 734, 0.40, oi=0),
        {**_quote("PUT", "2026-11-19", 729, 0.50), "option_type": "put"},
        _quote("WRONG_DTE", "2026-11-23", 735, 0.30),
    ]

    payload = _build(quotes=quotes)

    assert payload["aggregate_decision"] == "abstain"
    assert payload["ladder"] == []
    assert payload["rejected_quote_count"] == len(quotes)


def test_after_1230_window_rejects_publication():
    payload = _build(now=datetime(2026, 11, 18, 13, 0, tzinfo=NY))

    assert payload["aggregate_decision"] == "reject"
    assert "after_1230_et" in payload["reviews"][-1]["reasons"]


def test_same_inputs_produce_same_content_hash():
    assert _build()["lead_id"] == _build()["lead_id"]
