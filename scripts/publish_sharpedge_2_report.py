#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = Path(os.getenv("OUTDIR", "outputs"))
MIN_N = int(os.getenv("MIN_N", "20"))

REPORT_MD = OUTDIR / "sharpedge_2_report.md"
CARD_TXT = OUTDIR / "sharpedge_2_latest_card.txt"
DISCORD_TXT = OUTDIR / "sharpedge_2_discord_summary.txt"

CRITICAL_TABLES = [
    "auction_expectancy_events",
    "conditional_expectancy_matrix",
    "regime_daily",
    "open_resolution_regime",
    "options_positioning_metrics",
    "signals_daily",
]


def table_exists(con, table):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def require_tables(con):
    missing = [t for t in CRITICAL_TABLES if not table_exists(con, t)]
    if missing:
        raise RuntimeError(f"Missing critical tables: {missing}")


def read_latest(con, table, date_col, symbol_col):
    c = cols(con, table)
    if date_col not in c or symbol_col not in c:
        raise RuntimeError(f"{table} missing {date_col}/{symbol_col}")

    q = f"""
    SELECT *
    FROM {table}
    WHERE {symbol_col}=?
    ORDER BY {date_col} DESC
    LIMIT 1
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    if df.empty:
        raise RuntimeError(f"{table} has no latest row for {SYMBOL}")
    return df.iloc[0].to_dict()


def pick(row, names, default="NA"):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return default


def pct(x):
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.1%}"
    except Exception:
        return "NA"


def num(x, digits=2):
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def safe_text(x):
    if x is None or pd.isna(x):
        return "NA"
    return str(x)


def load_matrix(con):
    df = pd.read_sql_query(
        """
        SELECT *
        FROM conditional_expectancy_matrix
        ORDER BY tradability_score DESC
        """,
        con,
    )
    if df.empty:
        raise RuntimeError("conditional_expectancy_matrix is empty")

    if "n" not in df.columns:
        raise RuntimeError("conditional_expectancy_matrix missing n column")

    return df


def match_score(row, state):
    score = 0

    checks = {
        "regime_id": state.get("regime_id"),
        "vol_state": state.get("vol_state"),
        "vol_trend_state": state.get("vol_trend_state"),
        "macro_state": state.get("macro_state"),
        "dp_state": state.get("dp_state"),
        "open_regime_label": state.get("open_regime_label"),
        "dealer_state_hint": state.get("dealer_state_hint"),
    }

    for col, val in checks.items():
        if col in row.index and val not in [None, "NA"] and str(row[col]) == str(val):
            score += 1

    return score


def classify_recommendation(best):
    if best is None:
        return "DO_NOTHING", "No supported historical match."

    n = int(best.get("n", 0))
    expectancy = float(best.get("expectancy", 0) or 0)
    tradability = float(best.get("tradability_score", 0) or 0)
    fill_rate = float(best.get("fill_rate", 0) or 0)
    mae = abs(float(best.get("avg_MAE_pct", 0) or 0))
    max_dd = abs(float(best.get("max_drawdown", 0) or 0))

    if n < MIN_N:
        return "DO_NOTHING", f"Low sample support: n={n}, minimum={MIN_N}."

    if expectancy <= 0 or fill_rate < 0.50:
        return "DO_NOTHING", "No positive supported edge."

    if tradability >= 80 and fill_rate >= 0.65 and mae <= 0.006 and max_dd <= 0.02:
        return "AGGRESSIVE", "Strong supported expectancy with controlled excursion."

    if tradability >= 60 and fill_rate >= 0.58:
        return "NORMAL", "Supported edge, but still requires execution discipline."

    if tradability >= 45 and fill_rate >= 0.52:
        return "PROBE", "Some edge support, but not enough for normal size."

    return "DO_NOTHING", "Tradability score too weak."


def build_state(regime, openr, opt, sig):
    return {
        "date": pick(regime, ["date"]),
        "regime_id": pick(regime, ["regime_id"]),
        "regime_label": pick(regime, ["regime_label"]),
        "vol_state": pick(regime, ["vol_state"]),
        "vol_trend_state": pick(regime, ["vol_trend_state"]),
        "macro_state": pick(regime, ["macro_state"]),
        "dp_state": pick(regime, ["dp_state"]),
        "compression_flag": pick(regime, ["compression_flag"]),
        "open_regime_label": pick(openr, ["open_regime_label"]),
        "regime_confidence": pick(openr, ["regime_confidence"]),
        "setup_dir": pick(openr, ["setup_dir"]),
        "dealer_state_hint": pick(opt, ["dealer_state_hint"]),
        "spot": pick(opt, ["spot"]),
        "atm_strike": pick(opt, ["atm_strike"]),
        "gamma_wall": pick(opt, ["gamma_wall_strike", "max_total_oi_strike"]),
        "max_call_oi_strike": pick(opt, ["max_call_oi_strike"]),
        "max_put_oi_strike": pick(opt, ["max_put_oi_strike"]),
        "pcr_oi": pick(opt, ["pcr_oi"]),
        "early_score": pick(sig, ["early_score", "readiness_score"]),
        "early_bucket": pick(sig, ["early_bucket"]),
        "trade_permission": pick(sig, ["trade_permission"]),
    }


def choose_matches(matrix, state):
    df = matrix.copy()
    df["_match_score"] = df.apply(lambda r: match_score(r, state), axis=1)

    supported = df[df["n"].fillna(0).astype(int) >= MIN_N].copy()
    low = df[df["n"].fillna(0).astype(int) < MIN_N].copy()

    ranked = supported.sort_values(
        ["_match_score", "tradability_score", "n"],
        ascending=[False, False, False],
    )

    low_ranked = low.sort_values(
        ["_match_score", "tradability_score", "n"],
        ascending=[False, False, False],
    )

    best = ranked.iloc[0].to_dict() if not ranked.empty else None
    return best, ranked.head(10), low_ranked.head(5)


def risk_label(best):
    if best is None:
        return "NA", "NA"

    failed_fill = float(best.get("failed_fill_rate", 0) or 0)
    stop_proxy = float(best.get("stop_out_rate_proxy", 0) or 0)
    max_dd = abs(float(best.get("max_drawdown", 0) or 0))

    squeeze = "LOW"
    continuation = "LOW"

    if failed_fill >= 0.35 or stop_proxy >= 0.35:
        continuation = "HIGH"
    elif failed_fill >= 0.22 or stop_proxy >= 0.22:
        continuation = "MEDIUM"

    if max_dd >= 0.03:
        squeeze = "HIGH"
    elif max_dd >= 0.015:
        squeeze = "MEDIUM"

    return squeeze, continuation


def lines_for_match(m):
    if m is None:
        return [
            "- Sample support: NONE",
            "- Expected fill probability: NA",
            "- Expected time-to-fill: NA",
            "- Expected MAE before fill: NA",
        ]

    q = "LOW_SAMPLE" if int(m.get("n", 0)) < MIN_N else "SUPPORTED"

    return [
        f"- Sample support: {q} / n={int(m.get('n', 0))}",
        f"- Expected fill probability: {pct(m.get('fill_rate'))}",
        f"- Direct fill rate: {pct(m.get('direct_fill_rate'))}",
        f"- Failed fill rate: {pct(m.get('failed_fill_rate'))}",
        f"- Expected time-to-fill: {num(m.get('avg_time_to_fill_minutes'))} minutes",
        f"- Median time-to-fill: {num(m.get('median_time_to_fill_minutes'))} minutes",
        f"- Expected MAE before fill: {pct(m.get('avg_MAE_pct'))}",
        f"- Expected MFE: {pct(m.get('avg_MFE_pct'))}",
        f"- Payoff ratio: {num(m.get('payoff_ratio'))}",
        f"- Expectancy: {num(m.get('expectancy'), 4)}",
        f"- Sortino: {num(m.get('sortino_ratio'))}",
        f"- Max drawdown: {pct(m.get('max_drawdown'))}",
        f"- Tradability score: {num(m.get('tradability_score'))}",
    ]


def write_outputs(state, best, top, low):
    recommendation, rec_reason = classify_recommendation(best)
    squeeze_risk, continuation_risk = risk_label(best)

    playbook = "NA"
    if best:
        playbook = safe_text(
            best.get("playbook_condition")
            or best.get("event_type")
            or best.get("fill_path_type")
            or "Best conditional matrix match"
        )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    md = []
    md.append(f"# SharpEdge 2.0 Auction Expectancy Report")
    md.append("")
    md.append(f"- Generated UTC: {now}")
    md.append(f"- Symbol: {SYMBOL}")
    md.append(f"- Minimum supported sample: {MIN_N}")
    md.append("")

    md.append("## Today’s Market State")
    md.append(f"- Date: {state['date']}")
    md.append(f"- Regime: {state['regime_id']} / {state['regime_label']}")
    md.append(f"- Vol state: {state['vol_state']} / trend: {state['vol_trend_state']}")
    md.append(f"- Macro state: {state['macro_state']}")
    md.append(f"- Dark pool state: {state['dp_state']}")
    md.append(f"- Open resolution: {state['open_regime_label']} confidence={num(state['regime_confidence'])}")
    md.append(f"- Setup direction: {state['setup_dir']}")
    md.append(f"- Dealer hint: {state['dealer_state_hint']}")
    md.append(f"- Spot: {num(state['spot'])} | ATM: {num(state['atm_strike'])}")
    md.append(f"- Gamma / OI wall: {num(state['gamma_wall'])}")
    md.append(f"- Call wall: {num(state['max_call_oi_strike'])} | Put wall: {num(state['max_put_oi_strike'])}")
    md.append(f"- PCR OI: {num(state['pcr_oi'])}")
    md.append(f"- Signal bucket: {state['early_bucket']} score={num(state['early_score'])}")
    md.append(f"- Trade permission: {state['trade_permission']}")
    md.append("")

    md.append("## Is The Current Gap-Fill Setup Tradable?")
    md.append(f"- Recommendation: **{recommendation}**")
    md.append(f"- Reason: {rec_reason}")
    md.append(f"- Squeeze risk: {squeeze_risk}")
    md.append(f"- Continuation risk: {continuation_risk}")
    md.append(f"- Best matching playbook condition: {playbook}")
    md.append("")
    md.extend(lines_for_match(best))
    md.append("")

    md.append("## Most Similar Supported Historical Paths")
    if top.empty:
        md.append("- No supported historical paths met the sample guard.")
    else:
        for i, (_, r) in enumerate(top.iterrows(), 1):
            md.append(
                f"{i}. match={int(r['_match_score'])} | "
                f"n={int(r['n'])} | "
                f"event={safe_text(r.get('event_type'))} | "
                f"path={safe_text(r.get('fill_path_type'))} | "
                f"regime={safe_text(r.get('regime_id'))} | "
                f"open={safe_text(r.get('open_regime_label'))} | "
                f"fill={pct(r.get('fill_rate'))} | "
                f"expectancy={num(r.get('expectancy'), 4)} | "
                f"score={num(r.get('tradability_score'))}"
            )

    md.append("")
    md.append("## Low-Sample Rows To Watch But Not Trust")
    if low.empty:
        md.append("- None.")
    else:
        for i, (_, r) in enumerate(low.iterrows(), 1):
            md.append(
                f"{i}. LOW_SAMPLE n={int(r['n'])} | "
                f"event={safe_text(r.get('event_type'))} | "
                f"path={safe_text(r.get('fill_path_type'))} | "
                f"fill={pct(r.get('fill_rate'))} | "
                f"expectancy={num(r.get('expectancy'), 4)} | "
                f"score={num(r.get('tradability_score'))}"
            )

    card = []
    card.append("SHARPEDGE 2.0 LATEST CARD")
    card.append(f"Date: {state['date']} | Symbol: {SYMBOL}")
    card.append(f"State: {state['regime_id']} | {state['open_regime_label']} | {state['early_bucket']}")
    card.append(f"Recommendation: {recommendation}")
    card.append(f"Reason: {rec_reason}")
    card.append(f"Fill Prob: {pct(best.get('fill_rate') if best else None)}")
    card.append(f"Time-to-fill: {num(best.get('avg_time_to_fill_minutes') if best else None)} min")
    card.append(f"MAE before fill: {pct(best.get('avg_MAE_pct') if best else None)}")
    card.append(f"Squeeze Risk: {squeeze_risk}")
    card.append(f"Continuation Risk: {continuation_risk}")
    card.append(f"Playbook: {playbook}")
    card.append(f"Sample: n={int(best.get('n', 0)) if best else 0} / min={MIN_N}")

    discord = (
        f"SharpEdge 2.0 | {SYMBOL} {state['date']}\n"
        f"State: {state['regime_id']} | {state['open_regime_label']} | {state['early_bucket']}\n"
        f"Rec: {recommendation} — {rec_reason}\n"
        f"Fill: {pct(best.get('fill_rate') if best else None)} | "
        f"TTF: {num(best.get('avg_time_to_fill_minutes') if best else None)}m | "
        f"MAE: {pct(best.get('avg_MAE_pct') if best else None)}\n"
        f"Squeeze: {squeeze_risk} | Continuation: {continuation_risk}\n"
        f"Sample: n={int(best.get('n', 0)) if best else 0}/min={MIN_N}"
    )
    discord = discord[:1900]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    CARD_TXT.write_text("\n".join(card) + "\n", encoding="utf-8")
    DISCORD_TXT.write_text(discord + "\n", encoding="utf-8")

    print(f"OK: wrote {REPORT_MD}")
    print(f"OK: wrote {CARD_TXT}")
    print(f"OK: wrote {DISCORD_TXT}")


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        require_tables(con)

        regime = read_latest(con, "regime_daily", "date", "symbol")
        openr = read_latest(con, "open_resolution_regime", "session_date", "underlying")
        opt = read_latest(con, "options_positioning_metrics", "snapshot_ts", "underlying")
        sig = read_latest(con, "signals_daily", "date", "symbol")

        state = build_state(regime, openr, opt, sig)
        matrix = load_matrix(con)
        best, top, low = choose_matches(matrix, state)

        write_outputs(state, best, top, low)

    finally:
        con.close()


if __name__ == "__main__":
    main()
