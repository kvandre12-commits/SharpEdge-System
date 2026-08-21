#!/usr/bin/env python3
"""
Classify auction_expectancy_events into path-aware gap-fill outcomes.

Goal:
Move beyond binary fill/no-fill classification and encode HOW
the auction resolved intraday.

Inputs:
- auction_expectancy_events
- intraday bars table (default: spy_bars_15m)

Outputs:
- Updated auction_expectancy_events table
- outputs/auction_fill_path_labels.csv

New columns:
- fill_path_type
- acceptance_behavior
- initiative_behavior
- partial_fill
- failed_fill
- continuation_after_fill
- squeeze_then_fill
- rotational_balance_then_fill
- direct_fill

Path labels:
- DIRECT_FILL
- SQUEEZE_THEN_FILL
- ROTATIONAL_BALANCE_THEN_FILL
- ACCEPTANCE_RECLAIM
- STANDARD_FILL                 (ordinary reversion fill; catch-all for filled days)
- PARTIAL_FILL_REJECT
- FAILED_FILL_CONTINUATION
- LIQUIDITY_VACUUM_CONTINUATION
- NO_GAP
- NO_INTRADAY_BARS            (cannot classify path; missing 15m coverage)
- UNCLASSIFIED
"""

import os
import sqlite3

import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")

OUTPUT_CSV = "outputs/auction_fill_path_labels.csv"

FAST_FILL_BARS = int(os.getenv("FAST_FILL_BARS", "3"))
LOW_ADVERSE_EXCURSION_PCT = float(
    os.getenv("LOW_ADVERSE_EXCURSION_PCT", "0.0025")
)
ROTATION_BAND_PCT = float(os.getenv("ROTATION_BAND_PCT", "0.002"))
LIQUIDITY_VACUUM_RANGE_PCT = float(
    os.getenv("LIQUIDITY_VACUUM_RANGE_PCT", "0.012")
)
MIN_PARTIAL_FILL_RATIO = float(os.getenv("MIN_PARTIAL_FILL_RATIO", "0.50"))
SQUEEZE_AWAY_RATIO = float(os.getenv("SQUEEZE_AWAY_RATIO", "0.35"))


def connect():
    return sqlite3.connect(DB_PATH)


def table_exists(con, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def existing_cols(con, table_name: str):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table_name})")}


def ensure_columns(con):
    required = {
        "fill_path_type": "TEXT",
        "acceptance_behavior": "TEXT",
        "initiative_behavior": "TEXT",
        "partial_fill": "INTEGER",
        "failed_fill": "INTEGER",
        "continuation_after_fill": "INTEGER",
        "squeeze_then_fill": "INTEGER",
        "rotational_balance_then_fill": "INTEGER",
        "direct_fill": "INTEGER",
    }

    cols = existing_cols(con, "auction_expectancy_events")

    for c, typ in required.items():
        if c not in cols:
            con.execute(
                f"""
                ALTER TABLE auction_expectancy_events
                ADD COLUMN {c} {typ}
                """
            )

    con.commit()


def load_events(con):
    q = """
    SELECT
      session_date,
      symbol,
      event_type,
      gap_pct,
      gap_direction,
      gap_fill_level,
      fill_completed,
      fill_ts,
      session_open,
      session_high,
      session_low,
      session_close,
      open_regime_label,
      failed_breakdown_open,
      accepted_breakdown_open
    FROM auction_expectancy_events
    WHERE symbol = ?
    ORDER BY session_date ASC
    """

    df = pd.read_sql_query(q, con, params=(SYMBOL,))

    if df.empty:
        raise RuntimeError(
            "auction_expectancy_events returned 0 rows."
        )

    return df


def load_intraday(con, session_date: str):
    q = f"""
    SELECT
      ts,
      open,
      high,
      low,
      close,
      volume
    FROM {BARS_TABLE}
    WHERE symbol = ?
      AND session_date = ?
    ORDER BY ts ASC
    """

    df = pd.read_sql_query(q, con, params=(SYMBOL, session_date))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df


def classify_row(r, bars: pd.DataFrame):

    default = {
        "fill_path_type": "UNCLASSIFIED",
        "acceptance_behavior": None,
        "initiative_behavior": None,
        "partial_fill": 0,
        "failed_fill": 0,
        "continuation_after_fill": 0,
        "squeeze_then_fill": 0,
        "rotational_balance_then_fill": 0,
        "direct_fill": 0,
    }

    gap_pct = r["gap_pct"]

    if pd.isna(gap_pct) or abs(gap_pct) < 1e-8:
        default["fill_path_type"] = "NO_GAP"
        return default

    if bars.empty:
        default["fill_path_type"] = "NO_INTRADAY_BARS"
        default["acceptance_behavior"] = "NO_PATH_DATA"
        default["initiative_behavior"] = "NO_PATH_DATA"
        return default

    gap_dir = str(r["gap_direction"]).upper()
    fill_level = float(r["gap_fill_level"])

    open_px = float(r["session_open"])
    high_px = float(r["session_high"])
    low_px = float(r["session_low"])
    close_px = float(r["session_close"])

    bars = bars.reset_index(drop=True)

    # Canonical fill truth comes from the builder (fill_completed / fill_ts),
    # which already excludes the opening bar. We deliberately do NOT re-derive
    # fill from session extremes -- that created a second, conflicting fill
    # truth in the same table.
    filled = int(r.get("fill_completed", 0) or 0) == 1
    fill_ts = r.get("fill_ts")

    if gap_dir == "UP":
        gap_size = abs(open_px - fill_level)
        mfe_toward_fill = open_px - low_px
        adverse_away = high_px - open_px
    else:
        gap_size = abs(fill_level - open_px)
        mfe_toward_fill = high_px - open_px
        adverse_away = open_px - low_px

    move_toward_fill = mfe_toward_fill / gap_size if gap_size > 0 else 0.0
    adverse_ratio = adverse_away / open_px if open_px > 0 else 0.0

    # Locate the fill bar by the canonical fill_ts (already after the opening
    # bar), so a small gap whose OPENING bar straddles fill_level is never
    # mislabeled a fast DIRECT_FILL.
    fill_bar_idx = None
    if filled and fill_ts is not None and not bars.empty:
        fts = pd.to_datetime(fill_ts, errors="coerce")
        if pd.notna(fts):
            after = bars.index[bars["ts"] >= fts]
            if len(after):
                fill_bar_idx = int(after[0])

    # ---- filled paths (most specific first) ----
    if (
        filled
        and fill_bar_idx is not None
        and fill_bar_idx <= FAST_FILL_BARS
        and adverse_ratio <= LOW_ADVERSE_EXCURSION_PCT
    ):
        default["fill_path_type"] = "DIRECT_FILL"
        default["direct_fill"] = 1
        default["acceptance_behavior"] = "FAST_ACCEPTANCE"
        default["initiative_behavior"] = "IMMEDIATE_REVERSION"
        return default

    if filled and adverse_ratio >= SQUEEZE_AWAY_RATIO * abs(gap_pct):
        default["fill_path_type"] = "SQUEEZE_THEN_FILL"
        default["squeeze_then_fill"] = 1
        default["initiative_behavior"] = "SQUEEZE_AWAY_FIRST"
        default["acceptance_behavior"] = "LATE_FILL_ACCEPTANCE"
        return default

    midpoint = (open_px + fill_level) / 2.0
    bars_near_mid = 0
    for _, b in bars.head(8).iterrows():
        if midpoint:
            dist = abs(float(b["close"]) - midpoint) / midpoint
            if dist <= ROTATION_BAND_PCT:
                bars_near_mid += 1

    if filled and bars_near_mid >= 4:
        default["fill_path_type"] = "ROTATIONAL_BALANCE_THEN_FILL"
        default["rotational_balance_then_fill"] = 1
        default["initiative_behavior"] = "ROTATIONAL_AUCTION"
        default["acceptance_behavior"] = "BALANCE_TO_ACCEPTANCE"
        return default

    open_regime = str(r.get("open_regime_label", "") or "")
    failed_breakdown_open = int(r.get("failed_breakdown_open", 0) or 0)

    if filled and (
        failed_breakdown_open == 1 or "FAILED_BREAKDOWN" in open_regime
    ):
        default["fill_path_type"] = "ACCEPTANCE_RECLAIM"
        default["acceptance_behavior"] = "FAILED_BREAKDOWN_RECLAIM"
        default["initiative_behavior"] = "REVERSAL_ACCEPTANCE"
        return default

    # Catch-all for ordinary reversion fills that match no sharper path.
    # Without this, every "normal" fill day fell through to UNCLASSIFIED
    # (~44% of history), starving the expectancy matrix of the most common
    # outcome. STANDARD_FILL keeps the filled/no-fill accounting honest.
    if filled:
        default["fill_path_type"] = "STANDARD_FILL"
        default["acceptance_behavior"] = "STANDARD_ACCEPTANCE"
        default["initiative_behavior"] = "NORMAL_REVERSION"
        return default

    # ---- no-fill paths (everything below is guaranteed not filled) ----
    if move_toward_fill >= MIN_PARTIAL_FILL_RATIO:
        default["fill_path_type"] = "PARTIAL_FILL_REJECT"
        default["partial_fill"] = 1
        default["acceptance_behavior"] = "REJECTED_BEFORE_FILL"
        default["initiative_behavior"] = "FAILED_REVERSION"
        return default

    if (gap_dir == "UP" and close_px > open_px) or (
        gap_dir == "DOWN" and close_px < open_px
    ):
        default["fill_path_type"] = "FAILED_FILL_CONTINUATION"
        default["failed_fill"] = 1
        default["continuation_after_fill"] = 1
        default["acceptance_behavior"] = "CONTINUATION_ACCEPTANCE"
        default["initiative_behavior"] = "TREND_CONTINUATION"

        range_pct = (high_px - low_px) / open_px if open_px else 0.0
        if range_pct >= LIQUIDITY_VACUUM_RANGE_PCT:
            default["fill_path_type"] = "LIQUIDITY_VACUUM_CONTINUATION"
            default["initiative_behavior"] = "LIQUIDITY_VACUUM"

        return default

    return default


def main():

    con = connect()

    try:

        if not table_exists(con, "auction_expectancy_events"):
            raise RuntimeError(
                "Missing auction_expectancy_events table."
            )

        if not table_exists(con, BARS_TABLE):
            raise RuntimeError(
                f"Missing intraday bars table: {BARS_TABLE}"
            )

        ensure_columns(con)

        events = load_events(con)

        out_rows = []

        for _, r in events.iterrows():

            session_date = r["session_date"]

            bars = load_intraday(con, session_date)

            cls = classify_row(r, bars)

            out_rows.append({
                "session_date": session_date,
                "symbol": SYMBOL,
                "event_type": r["event_type"],
                **cls,
            })

            con.execute(
                """
                UPDATE auction_expectancy_events
                SET
                  fill_path_type = ?,
                  acceptance_behavior = ?,
                  initiative_behavior = ?,
                  partial_fill = ?,
                  failed_fill = ?,
                  continuation_after_fill = ?,
                  squeeze_then_fill = ?,
                  rotational_balance_then_fill = ?,
                  direct_fill = ?
                WHERE symbol = ?
                  AND session_date = ?
                  AND event_type = ?
                """,
                (
                    cls["fill_path_type"],
                    cls["acceptance_behavior"],
                    cls["initiative_behavior"],
                    cls["partial_fill"],
                    cls["failed_fill"],
                    cls["continuation_after_fill"],
                    cls["squeeze_then_fill"],
                    cls["rotational_balance_then_fill"],
                    cls["direct_fill"],
                    SYMBOL,
                    session_date,
                    r["event_type"],
                ),
            )

        con.commit()

        os.makedirs("outputs", exist_ok=True)

        out = pd.DataFrame(out_rows)

        out.to_csv(OUTPUT_CSV, index=False)

        print(
            f"OK: classified fill paths rows={len(out)} "
            f"-> {OUTPUT_CSV}"
        )

        print(
            out["fill_path_type"]
            .value_counts(dropna=False)
            .to_string()
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
