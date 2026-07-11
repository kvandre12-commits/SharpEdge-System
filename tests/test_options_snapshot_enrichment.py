from __future__ import annotations

import sqlite3

from scripts.ingest_cboe_options_chain_snapshots import build_rows
from scripts.options_snapshot_store import ensure_table, upsert_row


def test_build_rows_includes_extended_cboe_fields():
    options = [
        {
            "option": "SPY260625C00100000",
            "open_interest": 120,
            "volume": 15,
            "gamma": 0.03,
            "iv": 0.19,
            "theta": -0.08,
            "vega": 0.11,
            "rho": 0.03,
            "theo": 1.31,
            "last_trade_price": 1.35,
            "bid": 1.2,
            "ask": 1.4,
        },
        {
            "option": "SPY260625P00100000",
            "open_interest": 80,
            "volume": 22,
            "gamma": 0.02,
            "iv": 0.21,
            "theta": -0.07,
            "vega": 0.1,
            "rho": -0.02,
            "theo": 1.21,
            "last_trade_price": 1.18,
            "bid": 1.1,
            "ask": 1.3,
        },
    ]

    rows = list(build_rows(options, "2026-06-25T14:01:26Z", "2026-06-25"))

    assert len(rows) == 1
    assert len(rows[0]) == 28
    assert rows[0][14] == -0.08
    assert rows[0][15] == -0.07
    assert rows[0][22] == 1.35
    assert rows[0][23] == 1.18
    assert rows[0][24] == 1.2
    assert rows[0][27] == 1.3


def test_upsert_row_accepts_extended_cboe_tuple_shape():
    con = sqlite3.connect(":memory:")
    ensure_table(con)
    row = (
        "2026-06-25T14:01:26Z",
        "2026-06-25",
        "SPY",
        "2026-06-25",
        0,
        100.0,
        120,
        80,
        15,
        22,
        0.03,
        0.02,
        0.19,
        0.21,
        -0.08,
        -0.07,
        0.11,
        0.1,
        0.03,
        -0.02,
        1.31,
        1.21,
        1.35,
        1.18,
        1.2,
        1.4,
        1.1,
        1.3,
    )

    upsert_row(con, row, source="cboe")
    stored = con.execute(
        "SELECT call_theta, put_theta, call_last_trade_price, put_last_trade_price, call_bid, put_ask FROM options_chain_snapshots"
    ).fetchone()

    assert stored == (-0.08, -0.07, 1.35, 1.18, 1.2, 1.3)
