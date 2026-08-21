from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from expectancy_engine import (  # noqa: E402
    NO_DTE_BUCKET,
    build_joined_df,
    compute_grid,
)


def test_compute_grid_all_days_includes_gated_rows_and_gated_subset():
    df = pd.DataFrame(
        [
            {
                "trade_gate": 1,
                "regime_label": "trend",
                "pressure_state": "NORMAL",
                "dte_bucket": "0DTE",
                "ret_1d": 0.02,
            },
            {
                "trade_gate": 0,
                "regime_label": "trend",
                "pressure_state": "NORMAL",
                "dte_bucket": "0DTE",
                "ret_1d": -0.01,
            },
        ]
    )

    grid = compute_grid(df)
    all_days = grid[grid["gate_view"] == "ALL_DAYS"].iloc[0]
    gated = grid[grid["gate_view"] == "GATED_ONLY"].iloc[0]

    assert int(all_days["n"]) == 2
    assert float(all_days["expectancy"]) == 0.005
    assert int(gated["n"]) == 1
    assert float(gated["expectancy"]) == 0.02


def test_build_joined_df_reads_session_date_dte_and_dedupes_daily_joins():
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(
            """
            CREATE TABLE signals_daily (
                date TEXT,
                symbol TEXT,
                trade_gate INTEGER,
                pressure_state TEXT
            );
            CREATE TABLE features_daily (
                date TEXT,
                symbol TEXT,
                ret_1d REAL
            );
            CREATE TABLE regime_daily (
                date TEXT,
                symbol TEXT,
                regime_label TEXT
            );
            CREATE TABLE execution_state_daily (
                session_date TEXT,
                symbol TEXT,
                dte_bucket TEXT
            );
            """
        )
        con.executemany(
            "INSERT INTO signals_daily VALUES (?, ?, ?, ?)",
            [
                ("2026-01-02", "SPY", 1, "NORMAL"),
                ("2026-01-03", "SPY", 0, "COILED"),
            ],
        )
        con.executemany(
            "INSERT INTO features_daily VALUES (?, ?, ?)",
            [
                ("2026-01-02", "SPY", 0.01),
                ("2026-01-03", "SPY", -0.02),
            ],
        )
        con.executemany(
            "INSERT INTO regime_daily VALUES (?, ?, ?)",
            [
                ("2026-01-02", "SPY", "old_duplicate"),
                ("2026-01-02", "SPY", "trend"),
                ("2026-01-03", "SPY", "balance"),
            ],
        )
        con.executemany(
            "INSERT INTO execution_state_daily VALUES (?, ?, ?)",
            [
                ("2026-01-02", "SPY", "0DTE"),
                ("2026-01-03", "SPY", "1DTE"),
            ],
        )

        df = build_joined_df(con).sort_values("date").reset_index(drop=True)

        assert len(df) == 2
        assert df.loc[0, "regime_label"] == "trend"
        assert df["dte_bucket"].tolist() == ["0DTE", "1DTE"]
    finally:
        con.close()


def test_build_joined_df_uses_explicit_no_dte_label_when_no_source_exists():
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(
            """
            CREATE TABLE signals_daily (
                date TEXT,
                symbol TEXT,
                trade_gate INTEGER,
                pressure_state TEXT
            );
            CREATE TABLE features_daily (
                date TEXT,
                symbol TEXT,
                ret_1d REAL
            );
            """
        )
        con.execute(
            "INSERT INTO signals_daily VALUES (?, ?, ?, ?)",
            ("2026-01-02", "SPY", 0, "NORMAL"),
        )
        con.execute(
            "INSERT INTO features_daily VALUES (?, ?, ?)",
            ("2026-01-02", "SPY", 0.01),
        )

        df = build_joined_df(con)

        assert df["dte_bucket"].tolist() == [NO_DTE_BUCKET]
    finally:
        con.close()
