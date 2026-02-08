#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = os.getenv("OUTDIR", "outputs")

SUMMARY_MD = os.path.join(OUTDIR, "results_summary.md")
PRESSURE_CSV = os.path.join(OUTDIR, "edge_by_pressure_state.csv")
EQUITY_PNG = os.path.join(OUTDIR, f"{SYMBOL.lower()}_trade_gate_equity_curve.png")


def safe_sharpe(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.sqrt(252) * float(r.mean()) / sd)


def load(con: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT
      s.date,
      s.symbol,
      s.trade_gate,
      s.pressure_state,
      f.ret_1d
    FROM signals_daily s
    JOIN features_daily f
      ON f.symbol = s.symbol
     AND f.date = s.date
    WHERE s.symbol = ?
    ORDER BY s.date
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    if df.empty:
        raise RuntimeError("0 rows from signals_daily ⋈ features_daily. Check tables and date alignment.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["trade_gate"] = pd.to_numeric(df["trade_gate"], errors="coerce").fillna(0).astype(int)
    df["pressure_state"] = df["pressure_state"].astype("string")
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    return df


def metrics(returns: pd.Series) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) == 0:
        return {"n": 0, "win_rate": float("nan"), "expectancy": float("nan"),
                "sharpe": float("nan"), "max_dd": float("nan"), "equity": None}
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "n": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "sharpe": float(safe_sharpe(r)),
        "max_dd": float(dd.min()) if len(dd) else float("nan"),
        "equity": equity,
    }


def edge_by_pressure(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["pressure_state"] = t["pressure_state"].fillna("UNKNOWN").astype(str)

    rows = []
    for state, g in t.groupby("pressure_state", dropna=False):
        r = pd.to_numeric(g["ret_1d"], errors="coerce").dropna()
        if len(r) == 0:
            continue
        rows.append({
            "pressure_state": state,
            "n": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "expectancy": float(r.mean()),
            "sharpe": float(safe_sharpe(r)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["expectancy", "n"], ascending=[False, False]).reset_index(drop=True)


def plot_equity(equity: pd.Series):
    os.makedirs(OUTDIR, exist_ok=True)
    plt.figure()
    equity.reset_index(drop=True).plot()
    plt.title(f"Equity Curve — {SYMBOL} (trade_gate == 1)")
    plt.ylabel("Growth of $1")
    plt.xlabel("Trade #")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_PNG)
    plt.close()


def fmt(x, pct=False, digits=4):
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.2%}" if pct else f"{x:.{digits}f}"


def write_summary(df_all: pd.DataFrame, m_all: dict, m_gated: dict, m_ungated: dict):
    dmin = df_all["date"].min()
    dmax = df_all["date"].max()

    lines = []
    lines.append(f"# Proof of Edge — trade_gate evaluation ({SYMBOL})\n")
    lines.append(f"- Date range: {dmin.date() if pd.notna(dmin) else 'NA'} → {dmax.date() if pd.notna(dmax) else 'NA'}")
    lines.append(f"- Total rows (signals ⋈ features): {len(df_all)}\n")

    lines.append("## Gated (trade_gate == 1)")
    lines.append(f"- Trades: {m_gated['n']}")
    lines.append(f"- Win rate: {fmt(m_gated['win_rate'], pct=True)}")
    lines.append(f"- Expectancy (mean ret_1d): {fmt(m_gated['expectancy'])}")
    lines.append(f"- Sharpe (ann): {fmt(m_gated['sharpe'], digits=2)}")
    lines.append(f"- Max drawdown: {fmt(m_gated['max_dd'], pct=True)}\n")

    lines.append("## Ungated baseline (trade_gate == 0)")
    lines.append(f"- Trades: {m_ungated['n']}")
    lines.append(f"- Win rate: {fmt(m_ungated['win_rate'], pct=True)}")
    lines.append(f"- Expectancy (mean ret_1d): {fmt(m_ungated['expectancy'])}")
    lines.append(f"- Sharpe (ann): {fmt(m_ungated['sharpe'], digits=2)}")
    lines.append(f"- Max drawdown: {fmt(m_ungated['max_dd'], pct=True)}\n")

    lines.append("## All days baseline (ignoring gate)")
    lines.append(f"- Trades: {m_all['n']}")
    lines.append(f"- Win rate: {fmt(m_all['win_rate'], pct=True)}")
    lines.append(f"- Expectancy (mean ret_1d): {fmt(m_all['expectancy'])}")
    lines.append(f"- Sharpe (ann): {fmt(m_all['sharpe'], digits=2)}")
    lines.append(f"- Max drawdown: {fmt(m_all['max_dd'], pct=True)}\n")

    lines.append("## Artifacts")
    lines.append(f"- {os.path.basename(SUMMARY_MD)}")
    lines.append(f"- {os.path.basename(PRESSURE_CSV)}")
    lines.append(f"- {os.path.basename(EQUITY_PNG)}\n")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    try:
        df = load(con)

        m_all = metrics(df["ret_1d"])
        df_g = df[df["trade_gate"] == 1].copy()
        df_u = df[df["trade_gate"] == 0].copy()

        m_g = metrics(df_g["ret_1d"])
        m_u = metrics(df_u["ret_1d"])

        # breakdown only on gated trades (your “system takes trades” subset)
        breakdown = edge_by_pressure(df_g)
        if not breakdown.empty:
            breakdown.to_csv(PRESSURE_CSV, index=False)
        else:
            # still create file so CI has deterministic outputs
            pd.DataFrame(columns=["pressure_state", "n", "win_rate", "expectancy", "sharpe"]).to_csv(PRESSURE_CSV, index=False)

        if m_g.get("equity") is not None:
            plot_equity(m_g["equity"])
        else:
            # deterministic placeholder: empty plot file is annoying; write a tiny note instead
            with open(EQUITY_PNG.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                f.write("No gated trades with usable ret_1d; equity curve not generated.\n")

        write_summary(df, m_all, m_g, m_u)

        print(f"OK: wrote {SUMMARY_MD}")
        print(f"OK: wrote {PRESSURE_CSV}")
        if os.path.exists(EQUITY_PNG):
            print(f"OK: wrote {EQUITY_PNG}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
