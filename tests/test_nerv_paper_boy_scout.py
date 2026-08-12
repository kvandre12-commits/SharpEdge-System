from __future__ import annotations

from datetime import UTC, datetime

from scripts.nerv.paper_boy_scout import build_scout_payload

NOW = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)


def _contract(symbol: str, score: float = 82.0, *, age: int = 60, flags: str = ""):
    return {
        "underlying": symbol,
        "contract_symbol": f"{symbol}260821C00100000",
        "option_type": "call",
        "expiration": "2026-08-21",
        "nerv_score": score,
        "manual_validation_priority": "high" if not flags else "reject",
        "rejection_flags": flags,
        "quote_age_seconds": age,
    }


def _board(*contracts):
    return {
        "schema": "sharpedge.nerv_liquidity_board.v1",
        "summary": {
            "source": "yfinance",
            "data_mode": "research_delayed_or_unknown",
            "fetch_timestamp": "2026-08-12T15:55:00+00:00",
            "requested_symbols": sorted({row["underlying"] for row in contracts}),
            "errors": [],
        },
        "contracts": list(contracts),
    }


def test_scout_nominates_fresh_multi_contract_symbol_and_excludes_existing_lane():
    board = _board(
        *[_contract("AMZN", score) for score in (88, 85, 82)],
        *[_contract("AAPL", score) for score in (95, 94, 93)],
    )
    catalyst = {"lanes": [{"symbol": "AAPL", "status": "active_existing_pilot"}]}

    payload = build_scout_payload(board, catalyst_universe=catalyst, as_of=NOW)

    assert payload["summary"]["nominated_symbols"] == ["AMZN"]
    rows = {row["symbol"]: row for row in payload["rows"]}
    assert rows["AMZN"]["state"] == "nominated"
    assert rows["AAPL"]["state"] == "existing_or_queued_lane"
    assert payload["governance"]["directional_output_allowed"] is False
    assert payload["governance"]["manifest_required_before_evidence"] is True


def test_scout_refuses_stale_snapshot_even_with_high_scores():
    board = _board(*[_contract("AMZN", 99) for _ in range(4)])
    board["summary"]["fetch_timestamp"] = "2026-08-12T12:00:00+00:00"

    payload = build_scout_payload(board, as_of=NOW)

    assert payload["summary"]["nominated_symbols"] == []
    assert payload["rows"][0]["state"] == "source_stale"


def test_scout_requires_non_rejected_contract_breadth():
    board = _board(
        _contract("MSFT", 90),
        _contract("MSFT", 90, flags="missing_midpoint"),
        _contract("MSFT", 90, age=5000),
    )

    payload = build_scout_payload(board, as_of=NOW)

    row = payload["rows"][0]
    assert row["usable_contract_count"] == 1
    assert row["state"] == "insufficient_usable_contracts"


def test_scout_rejects_unknown_board_schema():
    board = _board(_contract("AMZN"))
    board["schema"] = "mystery"

    try:
        build_scout_payload(board, as_of=NOW)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown schemas must fail closed")
