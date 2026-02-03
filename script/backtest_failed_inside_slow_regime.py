#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# ---- knobs: "favorable slow regime" definition ----
FAV_VOL_STATES = set(os.getenv("FAV_VOL_STATES", "low,mid").split(","))
FAV_VOL_TRENDS = set(os.getenv("FAV_VOL_TRENDS", "flat,falling").split(","))
FAV_DP_STATES  = set(os.getenv("FAV_DP_STATES", "normal,high").split(","))
FAV_REQUIRE_COMPRESSION = int(os.getenv("FAV_REQUIRE_COMPRESSION", "0"))  # 1 to require compression_flag=1

EVENT_TYPE = os.getenv("EVENT_TYPE", "FAILED_BREAKDOWN")  # can switch later


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        # Load daily closes (truth)
        bars = pd.read_sql_query(
            """
            SELECT date, symbol, close
            FROM bars_daily
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            con,
            params=(SYMBOL,),
        )
        if bars.empty:
            raise RuntimeError("bars_daily empty. Run ingest_spy_daily.py first.")
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.sort_values("date").reset_index(drop=True)

        # Build forward returns by trading-day index
        for k in [1, 2, 5]:
            bars[f"close_fwd_{k}"] = bars["close"].shift(-k)
            bars[f"ret_fwd_{k}"] = (bars[f"close_fwd_{k}"] / bars["close"]) - 1.0

        # Load events (fast regime)
        events = pd.read_sql_query(
            """
            SELECT
              session_date AS date,
              underlying AS symbol,
              regime_type,
              regime_confidence
            FROM liquidity_regime_events
            WHERE underlying = ?
              AND regime_type = ?
            ORDER BY session_date ASC
            """,
            con,
            params=(SYMBOL, EVENT_TYPE),
        )
        if events.empty:
            raise RuntimeError(f"No events found in liquidity_regime_events for {SYMBOL} / {EVENT_TYPE}.")
        events["date"] = pd.to_datetime(events["date"])

        # Load slow regimes
        slow = pd.read_sql_query(
            """
            SELECT
              date,
              symbol,
              vol_state,
              vol_trend_state,
              dp_state,
              compression_flag
            FROM regime_daily
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            con,
            params=(SYMBOL,),
        )
        if slow.empty:
            raise RuntimeError("regime_daily empty. Run build_regime_spy_daily.py first.")
        slow["date"] = pd.to_datetime(slow["date"])

        # Join: events + slow + bars(for returns)
        df = (
            events.merge(slow, on=["date", "symbol"], how="left")
                  .merge(bars[["date", "symbol", "close", "ret_fwd_1", "ret_fwd_2", "ret_fwd_5"]], on=["date", "symbol"], how="left")
        )

        # Favorable slow regime filter
        fav = (
            df["vol_state"].isin(FAV_VOL_STATES)
            & df["vol_trend_state"].isin(FAV_VOL_TRENDS)
            & df["dp_state"].isin(FAV_DP_STATES)
        )
        if FAV_REQUIRE_COMPRESSION == 1:
            fav = fav & (df["compression_flag"].fillna(0).astype(int) == 1)

        df["is_favorable_slow"] = fav.astype(int)

        # Helper: compute summary stats
        def summarize(x: pd.DataFrame, label: str) -> dict:
            out = {"bucket": label, "n": int(len(x))}
            for k in [1, 2, 5]:
                r = x[f"ret_fwd_{k}"].dropna()
                out[f"mean_fwd_{k}d"] = float(r.mean()) if len(r) else np.nan
                out[f"median_fwd_{k}d"] = float(r.median()) if len(r) else np.nan
                out[f"hit_rate_fwd_{k}d"] = float((r > 0).mean()) if len(r) else np.nan
            return out

        all_events = df.copy()
        fav_events = df[df["is_favorable_slow"] == 1].copy()

        summary = pd.DataFrame([
            summarize(all_events, f"ALL_{EVENT_TYPE}"),
            summarize(fav_events, f"FAV_SLOW_{EVENT_TYPE}")
        ])

        # Save outputs
        os.makedirs("outputs", exist_ok=True)
        df_out = df.sort_values("date")
        df_out.to_csv("outputs/failed_breakdown_backtest.csv", index=False)
        summary.to_csv("outputs/failed_breakdown_backtest_summary.csv", index=False)

        print("\n=== Backtest Summary ===")
        print(summary.to_string(index=False))
        print("\nWrote:")
        print(" - outputs/failed_breakdown_backtest.csv")
        print(" - outputs/failed_breakdown_backtest_summary.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
