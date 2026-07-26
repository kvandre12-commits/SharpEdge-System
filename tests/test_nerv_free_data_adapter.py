from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.io import write_liquidity_board_json, write_snapshot_csv  # noqa: E402
from nerv.models import NERVOptionQuote, NERVSnapshot  # noqa: E402
from nerv.scorer import build_liquidity_board, score_quote_record  # noqa: E402
from nerv.symbols import format_occ_symbol, parse_occ_symbol  # noqa: E402
from nerv.yfinance_adapter import YFinanceOptionsAdapter  # noqa: E402


def test_occ_symbol_round_trip() -> None:
    symbol = format_occ_symbol("SPY", "2026-01-16", "call", 500)

    assert symbol == "SPY260116C00500000"
    assert parse_occ_symbol(symbol) == {
        "underlying": "SPY",
        "expiration": "2026-01-16",
        "option_type": "call",
        "strike": 500.0,
        "contract_symbol": "SPY260116C00500000",
    }


def test_quote_record_adds_derived_fields() -> None:
    quote = NERVOptionQuote(
        underlying="SPY",
        contract_symbol="SPY260116C00500000",
        option_type="call",
        expiration="2026-01-16",
        strike=500,
        underlying_price=550,
        bid=10,
        ask=11,
        source="test",
        data_mode="unit",
        quote_timestamp="2026-01-01T00:00:00+00:00",
        fetch_timestamp="2026-01-01T00:01:30+00:00",
    )

    record = quote.to_record()

    assert record["midpoint"] == 10.5
    assert record["bid_ask_width"] == 1
    assert record["moneyness"] == round(500 / 550, 6)
    assert record["quote_age_seconds"] == 90
    assert "execution broker" in record["research_only_warning"]


@dataclass
class FakeChain:
    calls: pd.DataFrame
    puts: pd.DataFrame


class FakeTicker:
    options = ["2026-01-16", "2026-02-20"]
    fast_info = {"last_price": 550.0}

    def option_chain(self, expiration: str) -> FakeChain:
        assert expiration == "2026-01-16"
        return FakeChain(
            calls=pd.DataFrame(
                [
                    {
                        "contractSymbol": "SPY260116C00500000",
                        "lastTradeDate": "2026-01-01T00:00:00+00:00",
                        "strike": 500.0,
                        "lastPrice": 10.25,
                        "bid": 10.0,
                        "ask": 10.5,
                        "volume": 123,
                        "openInterest": 456,
                        "impliedVolatility": 0.22,
                        "inTheMoney": True,
                    }
                ]
            ),
            puts=pd.DataFrame(
                [
                    {
                        "contractSymbol": "SPY260116P00500000",
                        "lastTradeDate": "2026-01-01T00:00:00+00:00",
                        "strike": 500.0,
                        "lastPrice": 9.75,
                        "bid": 9.5,
                        "ask": 10.0,
                        "volume": 321,
                        "openInterest": 654,
                        "impliedVolatility": 0.24,
                        "inTheMoney": False,
                    }
                ]
            ),
        )


def test_scorer_prioritizes_tight_liquid_contract() -> None:
    strong = {
        "bid": 1.0,
        "ask": 1.05,
        "midpoint": 1.025,
        "volume": 700,
        "open_interest": 1400,
        "quote_age_seconds": 300,
    }
    weak = {
        "bid": 0.0,
        "ask": 0.0,
        "midpoint": None,
        "volume": 0,
        "open_interest": 1,
        "quote_age_seconds": 200000,
    }

    strong_score = score_quote_record(strong)
    weak_score = score_quote_record(weak)

    assert strong_score["manual_validation_priority"] == "high"
    assert strong_score["nerv_score"] > weak_score["nerv_score"]
    assert weak_score["manual_validation_priority"] == "reject"
    assert "stale_quote" in weak_score["rejection_flags"]


def test_scorer_marks_stale_liquid_contract_for_refresh() -> None:
    stale_liquid = {
        "bid": 1.0,
        "ask": 1.05,
        "midpoint": 1.025,
        "volume": 700,
        "open_interest": 1400,
        "quote_age_seconds": 200000,
    }

    score = score_quote_record(stale_liquid)

    assert score["manual_validation_priority"] == "refresh"
    assert score["nerv_score"] == 49.0
    assert score["fresh_quote_required"] is True


def test_liquidity_board_sorts_by_nerv_score() -> None:
    records = [
        {
            "contract_symbol": "BAD",
            "bid": 0.01,
            "ask": 0.5,
            "midpoint": 0.255,
            "volume": 0,
            "open_interest": 1,
            "quote_age_seconds": 200000,
        },
        {
            "contract_symbol": "GOOD",
            "bid": 1.0,
            "ask": 1.05,
            "midpoint": 1.025,
            "volume": 700,
            "open_interest": 1400,
            "quote_age_seconds": 300,
        },
    ]

    board = build_liquidity_board(records)

    assert board[0]["contract_symbol"] == "GOOD"
    assert board[0]["nerv_score"] > board[1]["nerv_score"]


def test_writers_include_score_fields(tmp_path: Path) -> None:
    quote = NERVOptionQuote(
        underlying="SPY",
        contract_symbol="SPY260116C00500000",
        option_type="call",
        expiration="2026-01-16",
        strike=500,
        source="test",
        data_mode="unit",
        fetch_timestamp="2026-01-01T00:05:00+00:00",
        quote_timestamp="2026-01-01T00:00:00+00:00",
        bid=1.0,
        ask=1.05,
        volume=700,
        open_interest=1400,
    )
    snapshot = NERVSnapshot(
        symbols=["SPY"],
        source="test",
        data_mode="unit",
        fetch_timestamp="2026-01-01T00:05:00+00:00",
        quotes=[quote],
    )

    csv_path = write_snapshot_csv(snapshot, tmp_path)
    board_path = write_liquidity_board_json(snapshot, tmp_path)

    assert "nerv_score" in csv_path.read_text()
    assert "manual_validation_priority" in board_path.read_text()


def test_yfinance_adapter_normalizes_fake_chain() -> None:
    adapter = YFinanceOptionsAdapter(ticker_factory=lambda _symbol: FakeTicker())

    snapshot = adapter.fetch(["spy"], expirations=["2026-01-16"])

    assert snapshot.summary()["quote_count"] == 2
    assert snapshot.errors == []
    call = snapshot.quotes[0].to_record()
    put = snapshot.quotes[1].to_record()
    assert call["underlying"] == "SPY"
    assert call["source"] == "yfinance"
    assert call["underlying_price"] == 550.0
    assert call["midpoint"] == 10.25
    assert put["option_type"] == "put"
    assert put["open_interest"] == 654
