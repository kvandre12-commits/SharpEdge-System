#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = os.getenv("OUTDIR", "outputs")

MIN_N = int(os.getenv("EDGE_MIN_N", "25"))          # minimum samples per bucket
ANNUALIZE = int(os.getenv("EDGE_ANNUALIZE", "252")) # daily annualization factor

OUT_CSV = os.path.join(OUTDIR, "edge_regime_pressure_dte.csv")
OUT_MD  = os.path.join(OUTDIR, "edge_regime_pressure_dte_top.md")


# -------------------------
# helpers
# -------------------------
def q1(con: sqlite3.Connection, sql: str, params=()) -> Optional[tuple]:
    cur = con.execute(sql, params)
    return cur.fetchone()

def table_exists(con: sqlite3.Connection, name: str) -> bool:
    r = q1(con, "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return r is not None

def cols(con: sqlite3.Connection, name: str) -> List[str]:
    # works for tables; views may return empty in older sqlite builds
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]
    except Exception:
        return []

def pick_first_existing(con: sqlite3.Connection, candidates: List[str]) -> Optional[str]:
    for t in candidates:
        if table_exists(con, t):
            return t
    return None

def safe_sharpe(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.sqrt(ANNUALIZE) * float(r.mean()) / sd)

def t_stat(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(mu / (sd / np.sqrt(len(r))))

def max_drawdown_from_returns(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())

def fail(msg: str):
    raise SystemExit(f"\nEXPECTANCY ENGINE FAIL: {msg}\n")


# -------------------------
# core: build joined frame
# -------------------------
def build_joined_df(con: sqlite3.Connection) -> pd.DataFrame:
    # Required base tables
    if not table_exists(con, "signals_daily"):
        fail("Missing table: signals_daily")
    if not table_exists(con, "features_daily"):
        fail("Missing table: features_daily")

    # Regime table naming mismatch (your common wiring bug)
    regime_table = pick_first_existing(con, ["spy_regime_daily", "regime_daily"])
    if regime_table is None:
        # Allow running without it, but label will be UNKNOWN
        regime_table = None

    # Liquidity regimes are often in liquidity_regime_events
    liq_table = "liquidity_regime_events" if table_exists(con, "liquidity_regime_events") else None

    # DTE bucket could live in multiple places depending on your build phase
    exec_table = "execution_state_daily" if table_exists(con, "execution_state_daily") else None
    trade_signals_table = "trade_signals" if table_exists(con, "trade_signals") else None

    # Figure out where DTE bucket exists
    dte_source = None
    dte_col = None

    # 1) execution_state_daily
    if exec_table:
        c = cols(con, exec_table)
        for cand in ["dte_bucket", "dte_recommendation", "dte_choice", "dte_selected"]:
            if cand in c:
                dte_source = exec_table
                dte_col = cand
                break

    # 2) trade_signals (if you persist)
    if dte_source is None and trade_signals_table:
        c = cols(con, trade_signals_table)
        if "dte_bucket" in c:
            dte_source = trade_signals_table
            dte_col = "dte_bucket"

    # 3) signals_daily (fallback)
    if dte_source is None:
        c = cols(con, "signals_daily")
        for cand in ["dte_bucket", "dte_recommendation", "dte_choice", "dte_selected"]:
            if cand in c:
                dte_source = "signals_daily"
                dte_col = cand
                break

    # Build query pieces
    # Base: signals_daily ⋈ features_daily
    # We always compute expectancy on forward return ret_1d (features_daily)
    # and slice by (regime_label, pressure_state, dte_bucket)
    join_sql = """
    WITH base AS (
      SELECT
        s.date AS date,
        s.symbol AS symbol,
        s.trade_gate AS trade_gate,
        s.pressure_state AS pressure_state,
        f.ret_1d AS ret_1d
      FROM signals_daily s
      JOIN features_daily f
        ON f.date = s.date
       AND f.symbol = s.symbol
      WHERE s.symbol = ?
    )
    SELECT
      b.date,
      b.symbol,
      b.trade_gate,
      b.pressure_state,
      b.ret_1d
    FROM base b
    ORDER BY b.date
    """

    df = pd.read_sql_query(join_sql, con, params=(SYMBOL,))
    if df.empty:
        fail("0 rows from signals_daily ⋈ features_daily. Check date alignment and that ret_1d exists.")

    # Normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["trade_gate"] = pd.to_numeric(df["trade_gate"], errors="coerce").fillna(0).astype(int)
    df["pressure_state"] = df["pressure_state"].astype("string").fillna("UNKNOWN")
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")

    # Add regime label if available
    if regime_table:
        rcols = cols(con, regime_table)
        # pick a label column
        label_col = "regime_label" if "regime_label" in rcols else None
        if label_col:
            rsql = f"SELECT date, symbol, {label_col} AS regime_label FROM {regime_table} WHERE symbol=?"
            rdf = pd.read_sql_query(rsql, con, params=(SYMBOL,))
            if not rdf.empty:
                rdf["date"] = pd.to_datetime(rdf["date"], errors="coerce")
                df = df.merge(rdf[["date","symbol","regime_label"]], on=["date","symbol"], how="left")
        else:
            df["regime_label"] = None
    else:
        df["regime_label"] = None

    df["regime_label"] = df.get("regime_label", pd.Series([None]*len(df))).fillna("UNKNOWN").astype(str)

    # Add liquidity regime type if available (optional slice later)
    if liq_table:
        lcols = cols(con, liq_table)
        # attempt to find the correct date key
        date_key = "session_date" if "session_date" in lcols else ("date" if "date" in lcols else None)
        sym_key  = "underlying" if "underlying" in lcols else ("symbol" if "symbol" in lcols else None)
        reg_col  = "regime_type" if "regime_type" in lcols else None

        if date_key and sym_key and reg_col:
            lsql = f"""
              SELECT
                {date_key} AS date,
                {sym_key} AS symbol,
                {reg_col} AS liquidity_regime
              FROM {liq_table}
              WHERE {sym_key} = ?
            """
            ldf = pd.read_sql_query(lsql, con, params=(SYMBOL,))
            if not ldf.empty:
                ldf["date"] = pd.to_datetime(ldf["date"], errors="coerce")
                df = df.merge(ldf[["date","symbol","liquidity_regime"]], on=["date","symbol"], how="left")

    df["liquidity_regime"] = df.get("liquidity_regime", pd.Series([None]*len(df))).fillna("UNKNOWN").astype(str)

    # Add DTE bucket if possible
    df["dte_bucket"] = "NA"
    if dte_source and dte_col:
        try:
            if dte_source in ("signals_daily",):
                dsql = f"SELECT date, symbol, {dte_col} AS dte_bucket FROM {dte_source} WHERE symbol=?"
                ddf = pd.read_sql_query(dsql, con, params=(SYMBOL,))
                ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
                df = df.merge(ddf[["date","symbol","dte_bucket"]], on=["date","symbol"], how="left", suffixes=("","_d"))
                df["dte_bucket"] = df["dte_bucket"].fillna("NA").astype(str)

            elif dte_source == "execution_state_daily":
                dsql = f"SELECT date, symbol, {dte_col} AS dte_bucket FROM execution_state_daily WHERE symbol=?"
                ddf = pd.read_sql_query(dsql, con, params=(SYMBOL,))
                ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
                df = df.merge(ddf[["date","symbol","dte_bucket"]], on=["date","symbol"], how="left", suffixes=("","_d"))
                df["dte_bucket"] = df["dte_bucket"].fillna("NA").astype(str)

            elif dte_source == "trade_signals":
                # trade_signals has entry_ts; map to session_date(date)
                dsql = """
                  SELECT
                    substr(entry_ts,1,10) AS date,
                    symbol,
                    dte_bucket
                  FROM trade_signals
                  WHERE symbol = ?
                """
                ddf = pd.read_sql_query(dsql, con, params=(SYMBOL,))
                ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
                df = df.merge(ddf[["date","symbol","dte_bucket"]], on=["date","symbol"], how="left", suffixes=("","_d"))
                df["dte_bucket"] = df["dte_bucket"].fillna("NA").astype(str)

        except Exception:
            # keep NA if join fails; better to run than die here
            pass

    # Optional: filter to gated days only if you want “system trades”
    # For discovery, we usually look at BOTH:
    # - trade_gate==1 subset
    # - and full baseline
    return df


# -------------------------
# compute expectancy grid
# -------------------------
def compute_grid(df: pd.DataFrame) -> pd.DataFrame:
    # We compute on ret_1d; you can swap to another horizon later
    base = df.copy()
    base["ret_1d"] = pd.to_numeric(base["ret_1d"], errors="coerce")

    # Build both views: gated-only + all-days
    base["gate_view"] = np.where(base["trade_gate"] == 1, "GATED_ONLY", "ALL_DAYS")

    groups = ["gate_view", "regime_label", "pressure_state", "dte_bucket"]

    rows = []
    for keys, g in base.groupby(groups, dropna=False):
        r = pd.to_numeric(g["ret_1d"], errors="coerce").dropna()
        if len(r) == 0:
            continue
        rows.append({
            "gate_view": keys[0],
            "regime_label": keys[1],
            "pressure_state": keys[2],
            "dte_bucket": keys[3],
            "n": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "expectancy": float(r.mean()),
            "sharpe_ann": float(safe_sharpe(r)),
            "t_stat": float(t_stat(r)),
            "max_dd": float(max_drawdown_from_returns(r)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        fail("No groups produced. Likely ret_1d missing or all NULL.")

    # apply min_n to the “ranked” view, but still keep raw output
    out["meets_min_n"] = out["n"] >= MIN_N

    # rank by expectancy (primary), then n
    out = out.sort_values(["gate_view", "meets_min_n", "expectancy", "n"],
                          ascending=[True, False, False, False]).reset_index(drop=True)
    return out


def write_outputs(grid: pd.DataFrame):
    os.makedirs(OUTDIR, exist_ok=True)
    grid.to_csv(OUT_CSV, index=False)

    # Top table for quick reading
    top = grid[grid["meets_min_n"]].copy()
    if top.empty:
        # still write something deterministic
        top = grid.head(25).copy()

    # markdown
    def fmt_pct(x): return "NA" if not np.isfinite(x) else f"{x:.2%}"
    def fmt_num(x, d=4): return "NA" if not np.isfinite(x) else f"{x:.{d}f}"

    lines = []
    lines.append(f"# Regime × Pressure × DTE Expectancy ({SYMBOL})\n")
    lines.append(f"- MIN_N: {MIN_N}")
    lines.append(f"- Output CSV: {os.path.basename(OUT_CSV)}\n")
    lines.append("## Top buckets (ranked by expectancy, then n)\n")
    lines.append("| view | regime | pressure | dte | n | win | exp | sharpe | t | maxDD |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in top.head(30).iterrows():
        lines.append(
            f"| {r['gate_view']} | {r['regime_label']} | {r['pressure_state']} | {r['dte_bucket']} | "
            f"{int(r['n'])} | {fmt_pct(r['win_rate'])} | {fmt_num(r['expectancy'])} | "
            f"{fmt_num(r['sharpe_ann'],2)} | {fmt_num(r['t_stat'],2)} | {fmt_pct(r['max_dd'])} |"
        )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        df = build_joined_df(con)

        # Fail loudly on obvious “wiring drift”
        if df["ret_1d"].dropna().empty:
            fail("ret_1d is entirely NULL after join. features_daily.ret_1d not populated or date join mismatch.")

        grid = compute_grid(df)
        write_outputs(grid)

        print(f"OK: wrote {OUT_CSV}")
        print(f"OK: wrote {OUT_MD}")
        print("\nQuick sanity:")
        print(grid.head(10).to_string(index=False))

    finally:
        con.close()


if __name__ == "__main__":
    main()
