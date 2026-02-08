#!/usr/bin/env python3
import os
import argparse
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config / helpers
# ----------------------------

@dataclass
class Metrics:
    n: int
    win_rate: float
    expectancy: float
    sharpe_ann: float
    t_stat: float
    max_dd: float
    trade_freq: float  # fraction of days traded


def fail(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def ensure_cols(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{ctx}: missing columns: {missing}. Have: {list(df.columns)}")


def iso_date(s) -> str:
    return pd.to_datetime(s).date().isoformat()


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def ann_sharpe(x: np.ndarray, periods_per_year: int = 252) -> float:
    if len(x) < 2:
        return float("nan")
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def t_stat_mean(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return float("nan")
    return float(mu / (sd / np.sqrt(n)))


def compute_metrics(
    all_days: pd.DataFrame,
    traded: pd.Series,
    ret_col: str,
    cost_bps: float,
) -> Metrics:
    """
    all_days: all dates
    traded: boolean mask same index as all_days indicating trade days
    ret_col: forward return column (e.g. ret_1d)
    cost_bps: round-trip cost applied per trade day (bps of notional)
    """
    df = all_days.copy()
    ensure_cols(df, [ret_col], "compute_metrics input")

    # returns on trade days
    r = df.loc[traded, ret_col].astype(float).to_numpy()

    # apply per-trade cost (bps -> decimal)
    cost = cost_bps / 10000.0
    r_net = r - cost

    n = int(np.sum(~np.isnan(r_net)))
    if n == 0:
        return Metrics(
            n=0,
            win_rate=float("nan"),
            expectancy=float("nan"),
            sharpe_ann=float("nan"),
            t_stat=float("nan"),
            max_dd=float("nan"),
            trade_freq=float(np.mean(traded)) if len(traded) else float("nan"),
        )

    win_rate = float(np.mean(r_net > 0))
    expct = float(np.nanmean(r_net))
    sh = ann_sharpe(r_net)
    ts = t_stat_mean(r_net)

    # equity curve only over trade sequence (simple additive for diagnostics)
    eq = np.nancumsum(r_net)
    mdd = max_drawdown(eq)

    trade_freq = float(np.mean(traded)) if len(traded) else float("nan")

    return Metrics(
        n=n,
        win_rate=win_rate,
        expectancy=expct,
        sharpe_ann=sh,
        t_stat=ts,
        max_dd=mdd,
        trade_freq=trade_freq,
    )


# ----------------------------
# Gates
# ----------------------------

def build_gate_functions() -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """
    Return dict of gate_name -> function(df)->boolean Series.
    Assumes df has pressure_state.
    """
    def col_eq(col: str, val: str) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: (df[col] == val)

    def col_neq(col: str, val: str) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: (df[col] != val)

    # Add gates you care about here.
    return {
        "BASELINE_ALL_DAYS": lambda df: pd.Series(True, index=df.index),
        "PRESSURE_is_NORMAL": col_eq("pressure_state", "NORMAL"),
        "PRESSURE_is_COILED": col_eq("pressure_state", "COILED"),
        "PRESSURE_not_UNRESOLVED": col_neq("pressure_state", "UNRESOLVED_PRESSURE"),
        "PRESSURE_NORMAL_or_COILED": lambda df: df["pressure_state"].isin(["NORMAL", "COILED"]),
    }


# ----------------------------
# Data load
# ----------------------------

def load_joined(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """
    Needs:
      signals_daily(date, symbol, pressure_state, trade_gate?)
      features_daily(date, symbol, ret_1d)

    We compute gates from pressure_state for OOS.
    """
    q = """
    SELECT
      s.date AS date,
      s.symbol AS symbol,
      s.pressure_state AS pressure_state,
      f.ret_1d AS ret_1d
    FROM signals_daily s
    JOIN features_daily f
      ON f.symbol = s.symbol
     AND f.date = s.date
    WHERE s.symbol = ?
    ORDER BY s.date
    """
    df = pd.read_sql_query(q, con, params=(symbol,))
    if df.empty:
        fail("Joined query returned 0 rows. Ensure signals_daily and features_daily overlap on (symbol,date).")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        fail(f"Found invalid date values after parsing. Example rows:\n{bad}")

    # loud contracts
    ensure_cols(df, ["symbol", "pressure_state", "ret_1d"], "load_joined result")
    if df["pressure_state"].isna().any():
        n = int(df["pressure_state"].isna().sum())
        fail(f"pressure_state has {n} NULLs. Fix upstream signals_daily generation before OOS.")

    return df


# ----------------------------
# Split OOS
# ----------------------------

def pick_best_gate_on_train(
    df_train: pd.DataFrame,
    gates: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    min_n: int,
    cost_bps: float,
) -> Tuple[str, pd.DataFrame]:
    rows = []
    for name, fn in gates.items():
        traded = fn(df_train)
        m = compute_metrics(df_train, traded, "ret_1d", cost_bps)
        rows.append({
            "gate_name": name,
            "n_trades": m.n,
            "trade_freq": m.trade_freq,
            "win_rate": m.win_rate,
            "expectancy": m.expectancy,
            "sharpe_ann": m.sharpe_ann,
            "t_stat": m.t_stat,
            "max_dd": m.max_dd,
            "meets_min_n": bool(m.n >= min_n),
        })
    res = pd.DataFrame(rows).sort_values(["meets_min_n", "expectancy"], ascending=[False, False])

    # pick: best expectancy among those meeting min_n; else best expectancy overall
    eligible = res[res["meets_min_n"] == True]
    if not eligible.empty:
        best = str(eligible.iloc[0]["gate_name"])
    else:
        best = str(res.iloc[0]["gate_name"])

    return best, res


def run_split_oos(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str],
    min_n: int,
    cost_bps: float,
    outdir: str,
) -> None:
    gates = build_gate_functions()

    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end) if test_end else None

    df_train = df[df["date"] <= train_end_dt].copy()
    df_test = df[df["date"] >= test_start_dt].copy()
    if test_end_dt is not None:
        df_test = df_test[df_test["date"] <= test_end_dt].copy()

    if df_train.empty:
        fail("Train split is empty. Check --train-end.")
    if df_test.empty:
        fail("Test split is empty. Check --test-start/--test-end.")

    best_gate, train_table = pick_best_gate_on_train(df_train, gates, min_n=min_n, cost_bps=cost_bps)

    # evaluate best gate + baseline on test
    baseline = gates["BASELINE_ALL_DAYS"]
    best_fn = gates[best_gate]

    m_base_train = compute_metrics(df_train, baseline(df_train), "ret_1d", cost_bps)
    m_best_train = compute_metrics(df_train, best_fn(df_train), "ret_1d", cost_bps)
    m_base_test = compute_metrics(df_test, baseline(df_test), "ret_1d", cost_bps)
    m_best_test = compute_metrics(df_test, best_fn(df_test), "ret_1d", cost_bps)

    os.makedirs(outdir, exist_ok=True)

    # details table (train ranking)
    details_path = os.path.join(outdir, "oos_split_details.csv")
    train_table.to_csv(details_path, index=False)

    # summary markdown (no tabulate dependency)
    summ_path = os.path.join(outdir, "oos_split_summary.md")

    def fmt(m: Metrics) -> str:
        return (
            f"n={m.n}, win_rate={m.win_rate:.3f}, exp={m.expectancy:.6f}, "
            f"sharpe_ann={m.sharpe_ann:.3f}, t_stat={m.t_stat:.3f}, "
            f"max_dd={m.max_dd:.4f}, trade_freq={m.trade_freq:.3f}"
        )

    with open(summ_path, "w", encoding="utf-8") as f:
        f.write("# OOS Split Validation\n\n")
        f.write(f"- train_end: {iso_date(train_end_dt)}\n")
        f.write(f"- test_start: {iso_date(test_start_dt)}\n")
        f.write(f"- test_end: {iso_date(test_end_dt) if test_end_dt is not None else 'END'}\n")
        f.write(f"- cost_bps: {cost_bps}\n")
        f.write(f"- min_n: {min_n}\n\n")

        f.write("## Selected gate (fit on train)\n\n")
        f.write(f"**{best_gate}**\n\n")

        f.write("## Train metrics\n\n")
        f.write(f"- Baseline: {fmt(m_base_train)}\n")
        f.write(f"- Selected: {fmt(m_best_train)}\n\n")

        f.write("## Test metrics (true OOS)\n\n")
        f.write(f"- Baseline: {fmt(m_base_test)}\n")
        f.write(f"- Selected: {fmt(m_best_test)}\n\n")

        # uplift on test
        if np.isfinite(m_base_test.expectancy) and np.isfinite(m_best_test.expectancy):
            uplift = m_best_test.expectancy - m_base_test.expectancy
            f.write(f"## Test uplift\n\n- expectancy_uplift: {uplift:.6f}\n")

    print(f"OK: wrote {details_path}")
    print(f"OK: wrote {summ_path}")
    print(f"OK: selected_gate={best_gate}")


# ----------------------------
# Walk-forward
# ----------------------------

def run_walkforward(
    df: pd.DataFrame,
    start: str,
    end: Optional[str],
    train_lookback_days: int,
    step_days: int,
    min_n: int,
    cost_bps: float,
    outdir: str,
) -> None:
    """
    For each decision date t:
      - Train window: (t - train_lookback_days ... t-1)
      - Pick best gate on that window
      - Apply it to next step_days period (t ... t+step_days-1) as "test"
    Produces per-window selections and an aggregate equity curve of applied decisions.
    """
    gates = build_gate_functions()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else df["date"].max()

    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    if df.empty:
        fail("Walk-forward range is empty. Check --wf-start/--wf-end.")

    df = df.sort_values("date").reset_index(drop=True)

    selections = []
    equity_rows = []

    # unique dates
    dates = pd.to_datetime(df["date"].dt.date.unique())
    dates = dates.sort_values()

    # step through calendar by index on unique trading dates
    i = 0
    while i < len(dates):
        t0 = dates[i]

        train_start = t0 - pd.Timedelta(days=train_lookback_days)
        train_end = t0 - pd.Timedelta(days=1)
        test_start = t0
        test_end = t0 + pd.Timedelta(days=step_days - 1)

        dtrain = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
        dtest = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(dtrain) < max(30, min_n):  # require some minimum history
            i += 1
            continue
        if dtest.empty:
            break

        best_gate, train_table = pick_best_gate_on_train(dtrain, gates, min_n=min_n, cost_bps=cost_bps)

        # apply selected gate to test window
        traded = gates[best_gate](dtest)
        r = dtest.loc[traded, "ret_1d"].astype(float).to_numpy() - (cost_bps / 10000.0)

        # record selection summary
        m_test = compute_metrics(dtest, gates[best_gate](dtest), "ret_1d", cost_bps)
        selections.append({
            "decision_date": iso_date(t0),
            "train_start": iso_date(train_start),
            "train_end": iso_date(train_end),
            "test_start": iso_date(test_start),
            "test_end": iso_date(test_end),
            "selected_gate": best_gate,
            "test_n": m_test.n,
            "test_expectancy": m_test.expectancy,
            "test_sharpe_ann": m_test.sharpe_ann,
            "test_trade_freq": m_test.trade_freq,
        })

        # equity curve contributions (timestamped by trade dates)
        # (simple additive equity on trade days)
        if len(r) > 0:
            trade_dates = dtest.loc[traded, "date"].dt.date.astype(str).tolist()
            for d, rv in zip(trade_dates, r):
                equity_rows.append({"date": d, "ret_1d_net": float(rv)})

        # advance by step_days worth of unique dates (approx)
        i += step_days

    if not selections:
        fail("Walk-forward produced 0 evaluation windows. Loosen parameters or expand date range.")

    os.makedirs(outdir, exist_ok=True)

    sel_df = pd.DataFrame(selections)
    sel_path = os.path.join(outdir, "walkforward_summary.csv")
    sel_df.to_csv(sel_path, index=False)

    eq_df = pd.DataFrame(equity_rows)
    if not eq_df.empty:
        eq_df = eq_df.sort_values("date")
        eq_df["equity"] = eq_df["ret_1d_net"].cumsum()
    eq_path = os.path.join(outdir, "walkforward_equity.csv")
    eq_df.to_csv(eq_path, index=False)

    print(f"OK: wrote {sel_path}")
    print(f"OK: wrote {eq_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "SPY"))
    ap.add_argument("--outdir", default=os.getenv("OUTDIR", "outputs"))
    ap.add_argument("--cost-bps", type=float, default=float(os.getenv("COST_BPS", "0.0")))
    ap.add_argument("--min-n", type=int, default=int(os.getenv("MIN_N", "50")))

    ap.add_argument("--mode", choices=["split", "walkforward"], default="split")

    # split
    ap.add_argument("--train-end", default=os.getenv("TRAIN_END", "2025-12-31"))
    ap.add_argument("--test-start", default=os.getenv("TEST_START", "2026-01-01"))
    ap.add_argument("--test-end", default=os.getenv("TEST_END", ""))

    # walkforward
    ap.add_argument("--wf-start", default=os.getenv("WF_START", "2024-01-01"))
    ap.add_argument("--wf-end", default=os.getenv("WF_END", ""))
    ap.add_argument("--wf-train-lookback-days", type=int, default=int(os.getenv("WF_TRAIN_LOOKBACK_DAYS", "252")))
    ap.add_argument("--wf-step-days", type=int, default=int(os.getenv("WF_STEP_DAYS", "21")))

    args = ap.parse_args()

    if not os.path.exists(args.db):
        fail(f"DB not found: {args.db}")

    con = sqlite3.connect(args.db)
    try:
        df = load_joined(con, args.symbol)
    finally:
        con.close()

    # normalize dates to day
    df["date"] = pd.to_datetime(df["date"].dt.date)

    if args.mode == "split":
        test_end = args.test_end.strip() or None
        run_split_oos(
            df=df,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=test_end,
            min_n=args.min_n,
            cost_bps=args.cost_bps,
            outdir=args.outdir,
        )
    else:
        wf_end = args.wf_end.strip() or None
        run_walkforward(
            df=df,
            start=args.wf_start,
            end=wf_end,
            train_lookback_days=args.wf_train_lookback_days,
            step_days=args.wf_step_days,
            min_n=args.min_n,
            cost_bps=args.cost_bps,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
