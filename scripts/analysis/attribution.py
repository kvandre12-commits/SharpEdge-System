#!/usr/bin/env python3
"""
Attribution for walkforward strategy equity.

Inputs (auto-detected):
- outputs/walkforward_equity.csv (date, ret_1d_net, equity)  <-- preferred
Fallback:
- DB join signals_daily + features_daily to build a gated return series.

Joins context from DB (if present):
- regime_daily: regime_label, vol_state, vol_trend_state, dp_state, macro_state, compression_flag, transition_label
- signals_daily: pressure_state, trade_gate
- (optionally) execution_state_daily: if it exists, will join any extra columns it finds

Outputs:
- outputs/attribution_daily.csv
- outputs/attribution_by_group.csv
- outputs/attribution_report.md

No optional deps (no tabulate, no matplotlib).
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def table_cols(con: sqlite3.Connection, name: str) -> List[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]


def safe_read_sql(con: sqlite3.Connection, q: str, params: Tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(q, con, params=params)
    except Exception:
        return pd.DataFrame()


def sharpe_ann(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return np.nan
    return (mu / sd) * np.sqrt(252.0)


def t_stat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return np.nan
    return mu / (sd / np.sqrt(n))


def max_drawdown(equity: np.ndarray) -> float:
    # equity is cumulative return series (not log)
    if len(equity) == 0:
        return np.nan
    peak = -np.inf
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return float(mdd)


def load_walkforward_equity(outputs_dir: Path) -> pd.DataFrame:
    wf_path = outputs_dir / "walkforward_equity.csv"
    if not wf_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(wf_path)
    # expected: date, ret_1d_net, equity
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # normalize column naming
    if "ret_1d_net" not in df.columns:
        # accept "ret" as fallback
        for c in ["ret", "ret_net", "ret_1d"]:
            if c in df.columns:
                df = df.rename(columns={c: "ret_1d_net"})
                break

    if "equity" not in df.columns:
        # compute if missing
        if "ret_1d_net" in df.columns:
            df["equity"] = df["ret_1d_net"].fillna(0.0).cumsum()

    if "ret_1d_net" not in df.columns:
        return pd.DataFrame()

    return df[["date", "ret_1d_net", "equity"]].copy()


def build_fallback_equity_from_db(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """
    Fallback: equity = ret_1d when trade_gate==1 else 0.
    Uses:
      - signals_daily(date,symbol,trade_gate,pressure_state)
      - features_daily(date,symbol,ret_1d)
    """
    if not (table_exists(con, "signals_daily") and table_exists(con, "features_daily")):
        return pd.DataFrame()

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
    df = safe_read_sql(con, q, (symbol,))
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["trade_gate"] = df["trade_gate"].fillna(0).astype(int)
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    df["ret_1d_net"] = np.where(df["trade_gate"] == 1, df["ret_1d"], 0.0)
    df["equity"] = df["ret_1d_net"].fillna(0.0).cumsum()
    return df[["date", "ret_1d_net", "equity", "trade_gate", "pressure_state"]].copy()


def join_context(con: sqlite3.Connection, base: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Joins regime_daily + signals_daily + execution_state_daily if present.
    Tries to be schema-tolerant.
    """
    out = base.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")

    # --- signals_daily context ---
    if table_exists(con, "signals_daily"):
        sig_cols = table_cols(con, "signals_daily")
        wanted = [c for c in ["date", "symbol", "trade_gate", "pressure_state"] if c in sig_cols]
        if "date" in wanted and "symbol" in wanted:
            q = f"""
            SELECT {",".join(wanted)}
            FROM signals_daily
            WHERE symbol = ?
            """
            s = safe_read_sql(con, q, (symbol,))
            if not s.empty:
                s["date"] = pd.to_datetime(s["date"], errors="coerce")
                out = out.merge(s.drop(columns=["symbol"], errors="ignore"), on="date", how="left")

    # --- regime_daily context ---
    if table_exists(con, "regime_daily"):
        reg_cols = table_cols(con, "regime_daily")
        wanted = [
            c for c in [
                "date","symbol",
                "regime_label",
                "vol_state","vol_trend_state",
                "dp_state","macro_state",
                "compression_flag",
                "transition_label",
            ]
            if c in reg_cols
        ]
        if "date" in wanted and "symbol" in wanted:
            q = f"""
            SELECT {",".join(wanted)}
            FROM regime_daily
            WHERE symbol = ?
            """
            r = safe_read_sql(con, q, (symbol,))
            if not r.empty:
                r["date"] = pd.to_datetime(r["date"], errors="coerce")
                out = out.merge(r.drop(columns=["symbol"], errors="ignore"), on="date", how="left")

    # --- execution_state_daily (optional, grab everything that isn't huge) ---
    if table_exists(con, "execution_state_daily"):
        ex_cols = table_cols(con, "execution_state_daily")
        # prefer joining on a date-like column
        join_col = None
        for c in ["date", "session_date"]:
            if c in ex_cols:
                join_col = c
                break
        if join_col:
            # keep a manageable subset (avoid blobby columns)
            keep = [c for c in ex_cols if c not in ["raw_json", "payload", "notes"]]
            if "symbol" in keep:
                q = f"SELECT {','.join(keep)} FROM execution_state_daily WHERE symbol = ?"
                e = safe_read_sql(con, q, (symbol,))
            else:
                q = f"SELECT {','.join(keep)} FROM execution_state_daily"
                e = safe_read_sql(con, q, ())
            if not e.empty:
                e[join_col] = pd.to_datetime(e[join_col], errors="coerce")
                e = e.dropna(subset=[join_col])
                if join_col != "date":
                    e = e.rename(columns={join_col: "date"})
                out = out.merge(e.drop(columns=["symbol"], errors="ignore"), on="date", how="left")

    return out


def group_stats(df: pd.DataFrame, group_cols: List[str], min_n: int) -> pd.DataFrame:
    rows = []
    # Use net returns for attribution (the thing you actually trade)
    rets = df["ret_1d_net"].to_numpy(dtype=float)

    # Build equity for drawdown by group (within-group equity, not global)
    for keys, g in df.groupby(group_cols, dropna=False):
        x = g["ret_1d_net"].to_numpy(dtype=float)
        n = int(np.sum(~np.isnan(x)))
        if n == 0:
            continue
        eq = np.nancumsum(np.nan_to_num(x, nan=0.0))
        rows.append({
            **{group_cols[i]: (keys[i] if isinstance(keys, tuple) else keys) for i in range(len(group_cols))},
            "n": n,
            "trade_freq": n / max(len(df), 1),
            "win_rate": float(np.mean(x[~np.isnan(x)] > 0)) if n else np.nan,
            "expectancy": float(np.nanmean(x)) if n else np.nan,
            "sharpe_ann": sharpe_ann(x),
            "t_stat": t_stat(x),
            "max_dd": max_drawdown(eq),
            "meets_min_n": bool(n >= min_n),
            "total_contribution": float(np.nansum(x)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sort by total contribution (who made the equity), then expectancy
    out = out.sort_values(["total_contribution", "expectancy"], ascending=[False, False])
    return out


def write_report(report_path: Path, by_group: pd.DataFrame, group_cols: List[str], min_n: int):
    lines = []
    lines.append("# Attribution Report")
    lines.append("")
    lines.append(f"- Grouping: `{', '.join(group_cols)}`")
    lines.append(f"- Min N per group: `{min_n}`")
    lines.append("")

    if by_group.empty:
        lines.append("No groups produced (missing data or empty returns).")
        report_path.write_text("\n".join(lines))
        return

    usable = by_group[by_group["meets_min_n"] == True].copy()
    if usable.empty:
        lines.append("No groups met min_n. Try lowering --min-n or collecting more history.")
        report_path.write_text("\n".join(lines))
        return

    top = usable.head(10)
    bot = usable.tail(10).sort_values("total_contribution", ascending=True)

    def fmt_df(d: pd.DataFrame) -> str:
        # Markdown table without tabulate
        cols = group_cols + ["n","expectancy","sharpe_ann","t_stat","max_dd","total_contribution"]
        d2 = d[cols].copy()
        for c in ["expectancy","sharpe_ann","t_stat","max_dd","total_contribution"]:
            d2[c] = d2[c].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        d2["n"] = d2["n"].astype(int).astype(str)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join(map(str, r)) + " |" for r in d2.values.tolist()]
        return "\n".join([header, sep] + rows)

    lines.append("## Top contributing groups")
    lines.append("")
    lines.append(fmt_df(top))
    lines.append("")
    lines.append("## Bottom contributing groups")
    lines.append("")
    lines.append(fmt_df(bot))
    lines.append("")
    report_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "SPY"))
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument(
        "--group-cols",
        default="regime_label,pressure_state",
        help="Comma-separated grouping columns for attribution (must exist after join).",
    )
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # 1) load walkforward equity if available
    wf = load_walkforward_equity(outputs_dir)
    if wf.empty:
        # 2) fallback to DB gated returns
        wf = build_fallback_equity_from_db(con, args.symbol)

    if wf.empty:
        raise SystemExit("No strategy equity found (no outputs/walkforward_equity.csv and DB fallback empty).")

    # 3) join context
    df = join_context(con, wf, args.symbol)

    # normalize some types
    df["ret_1d_net"] = pd.to_numeric(df["ret_1d_net"], errors="coerce")
    if "trade_gate" in df.columns:
        df["trade_gate"] = pd.to_numeric(df["trade_gate"], errors="coerce").fillna(0).astype(int)

    # 4) resolve grouping columns (only keep ones that exist)
    requested = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    group_cols = [c for c in requested if c in df.columns]
    if not group_cols:
        # always give something
        group_cols = ["pressure_state"] if "pressure_state" in df.columns else []

    # 5) write daily attribution
    daily_path = outputs_dir / "attribution_daily.csv"
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(daily_path, index=False)

    # 6) group stats
    by_group = group_stats(df, group_cols, args.min_n)
    by_group_path = outputs_dir / "attribution_by_group.csv"
    by_group.to_csv(by_group_path, index=False)

    # 7) report
    report_path = outputs_dir / "attribution_report.md"
    write_report(report_path, by_group, group_cols, args.min_n)

    con.close()
    print(f"OK: wrote {daily_path}")
    print(f"OK: wrote {by_group_path}")
    print(f"OK: wrote {report_path}")


if __name__ == "__main__":
    main()
