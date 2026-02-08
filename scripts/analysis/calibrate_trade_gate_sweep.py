#!/usr/bin/env python3
"""
Calibrate trade_gate scientifically:
- Loads daily returns (features_daily.ret_1d) + current model outputs (signals_daily.pressure_state)
- Optionally joins regime_daily.regime_label if present
- Sweeps candidate gates and reports:
    trade_freq, n_trades, expectancy, win_rate, sharpe_ann, t_stat, max_dd, uplift vs baseline
- Writes:
    outputs/gate_sweep_results.csv
    outputs/gate_sweep_top.md
Fails loudly if required tables/columns are missing.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = os.getenv("OUTDIR", "outputs")

MIN_TRADES_DEFAULT = int(os.getenv("GATE_MIN_TRADES", "30"))   # guardrail for statistical nonsense


# -------------------------
# helpers
# -------------------------

def require_cols(df: pd.DataFrame, cols: List[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[FAIL LOUD] Missing columns in {ctx}: {missing}. Have: {list(df.columns)}")


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def safe_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.sqrt(252) * float(r.mean()) / sd)


def t_stat_mean_zero(returns: pd.Series) -> float:
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.sqrt(n) * float(r.mean()) / sd)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if len(dd) else float("nan")


# -------------------------
# load data
# -------------------------

def load_base(con: sqlite3.Connection) -> pd.DataFrame:
    # signals_daily + features_daily
    if not table_exists(con, "signals_daily"):
        raise RuntimeError("[FAIL LOUD] Missing table: signals_daily")
    if not table_exists(con, "features_daily"):
        raise RuntimeError("[FAIL LOUD] Missing table: features_daily")

    q = """
    SELECT
      s.date AS date,
      s.symbol AS symbol,
      s.trade_gate AS trade_gate,
      s.pressure_state AS pressure_state,
      f.ret_1d AS ret_1d
    FROM signals_daily s
    JOIN features_daily f
      ON f.symbol = s.symbol
     AND f.date = s.date
    WHERE s.symbol = ?
    ORDER BY s.date
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    if df.empty:
        raise RuntimeError("[FAIL LOUD] Base join returned 0 rows. signals_daily/features_daily dates likely misaligned.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["trade_gate"] = pd.to_numeric(df["trade_gate"], errors="coerce").fillna(0).astype(int)
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    df["pressure_state"] = df["pressure_state"].fillna("UNKNOWN").astype(str)

    require_cols(df, ["date", "symbol", "trade_gate", "pressure_state", "ret_1d"], "base df")

    # Optional join regime_daily if present
    if table_exists(con, "regime_daily"):
        rq = """
        SELECT date, symbol, regime_label
        FROM regime_daily
        WHERE symbol = ?
        """
        reg = pd.read_sql_query(rq, con, params=(SYMBOL,))
        if not reg.empty and "regime_label" in reg.columns:
            reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
            reg["regime_label"] = reg["regime_label"].fillna("UNKNOWN").astype(str)
            df = df.merge(reg[["date", "symbol", "regime_label"]], on=["date", "symbol"], how="left")
        else:
            df["regime_label"] = "UNKNOWN"
    else:
        df["regime_label"] = "UNKNOWN"

    df["regime_label"] = df["regime_label"].fillna("UNKNOWN").astype(str)
    return df


# -------------------------
# gate definitions
# -------------------------

@dataclass
class GateDef:
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    notes: str


def build_gate_defs() -> List[GateDef]:
    # You can add/remove gates here without touching the evaluation logic.
    gates: List[GateDef] = []

    # 0) sanity / baseline
    gates.append(GateDef(
        name="BASELINE_ALL_DAYS",
        fn=lambda df: pd.Series(True, index=df.index),
        notes="No gating: every day"
    ))

    # 1) current trade_gate as-is (should show 0% right now)
    gates.append(GateDef(
        name="CURRENT_trade_gate_eq_1",
        fn=lambda df: df["trade_gate"] == 1,
        notes="What your pipeline currently allows"
    ))

    # 2) pressure-state only gates (adjust names to match your actual states)
    gates.append(GateDef(
        name="PRESSURE_not_UNRESOLVED",
        fn=lambda df: df["pressure_state"].str.upper().ne("UNRESOLVED_PRESSURE"),
        notes="Exclude only unresolved pressure"
    ))
    gates.append(GateDef(
        name="PRESSURE_is_NORMAL",
        fn=lambda df: df["pressure_state"].str.upper().eq("NORMAL"),
        notes="Only NORMAL pressure_state"
    ))
    gates.append(GateDef(
        name="PRESSURE_is_COILED",
        fn=lambda df: df["pressure_state"].str.upper().eq("COILED"),
        notes="Only COILED pressure_state"
    ))
    gates.append(GateDef(
        name="PRESSURE_NORMAL_or_COILED",
        fn=lambda df: df["pressure_state"].str.upper().isin(["NORMAL", "COILED"]),
        notes="NORMAL + COILED"
    ))

    # 3) regime-only gates (if regime_label exists)
    gates.append(GateDef(
        name="REGIME_not_UNKNOWN",
        fn=lambda df: df["regime_label"].str.upper().ne("UNKNOWN"),
        notes="Any day with a regime_label"
    ))

    # 4) combined gates (this is usually where edge appears)
    gates.append(GateDef(
        name="REGIME_COILED_and_PRESSURE_COILED",
        fn=lambda df: (df["regime_label"].str.upper().eq("COILED")) & (df["pressure_state"].str.upper().eq("COILED")),
        notes="High selectivity combo"
    ))
    gates.append(GateDef(
        name="REGIME_COILED_and_PRESSURE_NORMAL_or_COILED",
        fn=lambda df: (df["regime_label"].str.upper().eq("COILED")) & (df["pressure_state"].str.upper().isin(["NORMAL", "COILED"])),
        notes="Coiled regime; pressure not broken"
    ))

    return gates


# -------------------------
# evaluation
# -------------------------

def eval_gate(df: pd.DataFrame, mask: pd.Series) -> Dict[str, object]:
    m = mask.fillna(False).astype(bool)
    total_days = len(df)
    trade_days = int(m.sum())

    r_all = df["ret_1d"].astype(float)
    r = r_all[m].dropna()

    out: Dict[str, object] = {
        "total_days": total_days,
        "trade_days": trade_days,
        "trade_freq": (trade_days / total_days) if total_days else float("nan"),
        "n_trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "expectancy": float(r.mean()) if len(r) else float("nan"),
        "sharpe_ann": safe_sharpe(r) if len(r) else float("nan"),
        "t_stat": t_stat_mean_zero(r) if len(r) else float("nan"),
        "max_dd": max_drawdown_from_returns(r) if len(r) else float("nan"),
    }
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    try:
        df = load_base(con)
    finally:
        con.close()

    # Basic diagnostics (fail loudly if returns missing)
    if df["ret_1d"].isna().mean() > 0.05:
        raise RuntimeError(f"[FAIL LOUD] ret_1d has too many NULLs: {df['ret_1d'].isna().mean():.1%}")

    # Baseline metrics
    baseline = eval_gate(df, pd.Series(True, index=df.index))
    base_exp = baseline["expectancy"]
    base_sh = baseline["sharpe_ann"]

    # pressure_state inventory (helps you see spelling mismatches)
    pressure_counts = df["pressure_state"].value_counts(dropna=False).head(20)

    results = []
    for g in build_gate_defs():
        metrics = eval_gate(df, g.fn(df))
        metrics["gate_name"] = g.name
        metrics["notes"] = g.notes
        metrics["uplift_expectancy"] = (metrics["expectancy"] - base_exp) if np.isfinite(metrics["expectancy"]) and np.isfinite(base_exp) else float("nan")
        metrics["uplift_sharpe"] = (metrics["sharpe_ann"] - base_sh) if np.isfinite(metrics["sharpe_ann"]) and np.isfinite(base_sh) else float("nan")
        results.append(metrics)

    out = pd.DataFrame(results)

    # Add “usable” flag: enough trades + not NaN
    min_trades = MIN_TRADES_DEFAULT
    out["meets_min_trades"] = out["n_trades"] >= min_trades
    out["usable"] = out["meets_min_trades"] & out["expectancy"].apply(np.isfinite)

    # Rank: prioritize expectancy uplift, then sharpe uplift, then trade_freq closeness to target band
    # (we don't hardcode the band; we just sort smartly)
    out_sorted = out.sort_values(
        by=["usable", "uplift_expectancy", "uplift_sharpe", "trade_freq", "n_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    csv_path = os.path.join(OUTDIR, "gate_sweep_results.csv")
    out_sorted.to_csv(csv_path, index=False)

    # Write a top markdown summary (phone-friendly)
    md_path = os.path.join(OUTDIR, "gate_sweep_top.md")
    top = out_sorted.head(10).copy()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gate sweep (top 10)\n\n")
        f.write(f"- DB: `{DB_PATH}`\n")
        f.write(f"- Symbol: `{SYMBOL}`\n")
        f.write(f"- Days: {len(df)}\n")
        f.write(f"- Baseline expectancy (all days): {base_exp:.6f}\n")
        f.write(f"- Baseline sharpe_ann (all days): {base_sh:.3f}\n")
        f.write(f"- MIN_TRADES: {min_trades}\n\n")

        f.write("## pressure_state counts (top 20)\n\n")
        for k, v in pressure_counts.items():
            f.write(f"- {k}: {int(v)}\n")

        f.write("\n## Top gates\n\n")
        f.write(top.to_markdown(index=False))

    print("\n=== GATE SWEEP DONE ===")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    print("\nTop 5 gates:")
    cols = ["gate_name", "trade_freq", "n_trades", "expectancy", "sharpe_ann", "uplift_expectancy", "usable"]
    print(out_sorted[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
