from __future__ import annotations

import copy
from datetime import date, datetime, timedelta

import pytest

from scripts.alpha_swarm.data_steward import SNAPSHOT_SCHEMA
from scripts.alpha_swarm.lock_manifest import build_manifest
from scripts.alpha_swarm.options_expression_agent import OPTION_SNAPSHOT_SCHEMA
from scripts.alpha_swarm.snapshot_acquirer import (
    ACQUISITION_SCHEMA,
    acquire_research_snapshot,
    build_option_snapshot,
    build_research_snapshot,
    payload_sha256,
    write_once,
)


def _manifest():
    return build_manifest(
        run_id="acquirer-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _price_capture():
    start = datetime.fromisoformat("2026-08-10T10:10:00-04:00")
    bars = []
    for index in range(20):
        close = 100.0 + index * 0.02
        bars.append(
            {
                "timestamp": (start + timedelta(minutes=index)).isoformat(),
                "open": close - 0.01,
                "high": close + 0.03,
                "low": close - 0.03,
                "close": close,
                "volume": 100 if index < 15 else 150,
            }
        )
    return {
        "schema": ACQUISITION_SCHEMA,
        "provider": "fixture-price",
        "source_ref": "fixture://price/SPY/2026-08-10T10:29",
        "symbol": "SPY",
        "session_date": "2026-08-10",
        "latest_data_ts": "2026-08-10T10:29:00-04:00",
        "bars": bars,
    }


def _contract(index: int):
    strike = 95 + index * 0.5
    return {
        "contract_symbol": f"SPY-C-{index}",
        "option_type": "call",
        "expiration": "2026-08-21",
        "strike": strike,
        "bid": 1.0,
        "ask": 1.1,
        "quote_ts": "2026-08-10T10:29:00-04:00",
        "open_interest": 500,
        "volume": 50,
    }


def _options_capture(*, latest="2026-08-10T10:29:00-04:00"):
    return {
        "schema": ACQUISITION_SCHEMA,
        "provider": "fixture-options",
        "source_ref": f"fixture://options/SPY/{latest}",
        "symbol": "SPY",
        "session_date": "2026-08-10",
        "latest_data_ts": latest,
        "spot": 100.38,
        "contracts": [_contract(index) for index in range(100)],
    }


def _captured_at():
    return datetime.fromisoformat("2026-08-10T10:30:00-04:00")


def test_build_research_snapshot_is_neutral_hashed_and_paper_only():
    manifest = _manifest()
    price = _price_capture()
    options = _options_capture()
    snapshot = build_research_snapshot(
        manifest,
        slot_id=manifest["slots"][0]["slot_id"],
        captured_at=_captured_at(),
        price_capture=price,
        options_capture=options,
    )

    assert snapshot["schema"] == SNAPSHOT_SCHEMA
    assert set(snapshot["features"]) == {
        "spot",
        "vwap",
        "vs_vwap_pct",
        "momentum_15m_pct",
        "volume_ratio",
    }
    assert snapshot["features"]["volume_ratio"] == 1.5
    assert snapshot["price_source"]["source_sha256"] == payload_sha256(price)
    assert snapshot["options_source"]["source_sha256"] == payload_sha256(options)
    assert snapshot["paper_only"] is True
    assert snapshot["authoritative"] is False
    assert snapshot["execution_permitted"] is False


def test_compilation_does_not_mutate_raw_captures():
    manifest = _manifest()
    price = _price_capture()
    options = _options_capture()
    original = copy.deepcopy((price, options))
    build_research_snapshot(
        manifest,
        slot_id=manifest["slots"][0]["slot_id"],
        captured_at=_captured_at(),
        price_capture=price,
        options_capture=options,
    )
    assert (price, options) == original


def test_future_source_and_wrong_symbol_fail_closed():
    manifest = _manifest()
    options = _options_capture(latest="2026-08-10T10:31:00-04:00")
    with pytest.raises(ValueError, match="after captured_at"):
        build_research_snapshot(
            manifest,
            slot_id=manifest["slots"][0]["slot_id"],
            captured_at=_captured_at(),
            price_capture=_price_capture(),
            options_capture=options,
        )
    price = _price_capture()
    price["symbol"] = "QQQ"
    with pytest.raises(ValueError, match="symbol"):
        build_research_snapshot(
            manifest,
            slot_id=manifest["slots"][0]["slot_id"],
            captured_at=_captured_at(),
            price_capture=price,
            options_capture=_options_capture(),
        )


def test_thin_or_future_bars_fail_closed():
    manifest = _manifest()
    price = _price_capture()
    price["bars"] = price["bars"][:10]
    with pytest.raises(ValueError, match="at least 16"):
        build_research_snapshot(
            manifest,
            slot_id=manifest["slots"][0]["slot_id"],
            captured_at=_captured_at(),
            price_capture=price,
            options_capture=_options_capture(),
        )
    price = _price_capture()
    price["bars"][-1]["timestamp"] = "2026-08-10T10:31:00-04:00"
    with pytest.raises(ValueError, match="future-dated"):
        build_research_snapshot(
            manifest,
            slot_id=manifest["slots"][0]["slot_id"],
            captured_at=_captured_at(),
            price_capture=price,
            options_capture=_options_capture(),
        )


def test_build_option_snapshot_preserves_whole_chain_without_selection():
    manifest = _manifest()
    capture = _options_capture(latest="2026-08-10T10:45:00-04:00")
    for contract in capture["contracts"]:
        contract["quote_ts"] = "2026-08-10T10:45:00-04:00"
    snapshot = build_option_snapshot(
        manifest,
        slot_id=manifest["slots"][0]["slot_id"],
        captured_at=datetime.fromisoformat("2026-08-10T10:46:00-04:00"),
        options_capture=capture,
    )
    assert snapshot["schema"] == OPTION_SNAPSHOT_SCHEMA
    assert len(snapshot["contracts"]) == 100
    assert "expression" not in snapshot
    assert snapshot["source"]["source_sha256"] == payload_sha256(capture)


def test_injected_fetchers_receive_only_locked_symbol():
    manifest = _manifest()
    calls = []

    def price_fetcher(symbol):
        calls.append(("price", symbol))
        return _price_capture()

    def options_fetcher(symbol):
        calls.append(("options", symbol))
        return _options_capture()

    snapshot = acquire_research_snapshot(
        manifest,
        slot_id=manifest["slots"][0]["slot_id"],
        captured_at=_captured_at(),
        price_fetcher=price_fetcher,
        options_fetcher=options_fetcher,
    )
    assert calls == [("price", "SPY"), ("options", "SPY")]
    assert snapshot["symbol"] == "SPY"


def test_write_once_refuses_overwrite(tmp_path):
    target = tmp_path / "snapshot.json"
    write_once(target, {"one": 1})
    with pytest.raises(FileExistsError):
        write_once(target, {"two": 2})
