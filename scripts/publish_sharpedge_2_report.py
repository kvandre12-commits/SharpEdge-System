#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

RowDict = dict[str, Any]


def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def cols(con: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table_name})")}


def require_tables(con: sqlite3.Connection) -> None:
    missing = [
        table_name
        for table_name in CRITICAL_TABLES
        if not table_exists(con, table_name)
    ]
    if missing:
        raise RuntimeError(f"Missing critical tables: {missing}")


def fetch_one_dict(
    con: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
) -> RowDict | None:
    cur = con.execute(query, params)
    row = cur.fetchone()
    if row is None:
        return None
    column_names = [desc[0] for desc in cur.description]
    return dict(zip(column_names, row, strict=False))


def fetch_all_dicts(
    con: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
) -> list[RowDict]:
    cur = con.execute(query, params)
    column_names = [desc[0] for desc in cur.description]
    return [dict(zip(column_names, row, strict=False)) for row in cur.fetchall()]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pick(row: RowDict, names: list[str], default: Any = "NA") -> Any:
    for name in names:
        if name in row and not is_missing(row[name]):
            return row[name]
    return default


def pct(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.1%}"


def num(value: Any, digits: int = 2) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def safe_text(value: Any) -> str:
    if is_missing(value):
        return "NA"
    return str(value)


def read_latest(
    con: sqlite3.Connection, table_name: str, date_col: str, symbol_col: str
) -> RowDict:
    column_names = cols(con, table_name)
    if date_col not in column_names or symbol_col not in column_names:
        raise RuntimeError(f"{table_name} missing {date_col}/{symbol_col}")

    row = fetch_one_dict(
        con,
        f"""
        SELECT *
        FROM {table_name}
        WHERE {symbol_col}=?
        ORDER BY {date_col} DESC
        LIMIT 1
        """,
        (SYMBOL,),
    )
    if row is None:
        raise RuntimeError(f"{table_name} has no latest row for {SYMBOL}")
    return row


def load_matrix(con: sqlite3.Connection) -> list[RowDict]:
    rows = fetch_all_dicts(
        con,
        """
        SELECT *
        FROM conditional_expectancy_matrix
        ORDER BY tradability_score DESC
        """,
    )
    if not rows:
        raise RuntimeError("conditional_expectancy_matrix is empty")
    if "n" not in rows[0]:
        raise RuntimeError("conditional_expectancy_matrix missing n column")
    return rows


def match_score(row: RowDict, state: RowDict) -> int:
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
    for column, expected in checks.items():
        if expected in (None, "NA"):
            continue
        if str(row.get(column)) == str(expected):
            score += 1
    return score


def classify_recommendation(best: RowDict | None) -> tuple[str, str]:
    if best is None:
        return "DO_NOTHING", "No supported historical match."

    sample_n = to_int(best.get("n"), 0)
    expectancy = to_float(best.get("expectancy")) or 0.0
    tradability = to_float(best.get("tradability_score")) or 0.0
    fill_rate = to_float(best.get("fill_rate")) or 0.0
    mae = abs(to_float(best.get("avg_MAE_pct")) or 0.0)
    max_dd = abs(to_float(best.get("max_drawdown")) or 0.0)

    if sample_n < MIN_N:
        return "DO_NOTHING", f"Low sample support: n={sample_n}, minimum={MIN_N}."
    if expectancy <= 0 or fill_rate < 0.50:
        return "DO_NOTHING", "No positive supported edge."
    if tradability >= 80 and fill_rate >= 0.65 and mae <= 0.006 and max_dd <= 0.02:
        return "AGGRESSIVE", "Strong supported expectancy with controlled excursion."
    if tradability >= 60 and fill_rate >= 0.58:
        return "NORMAL", "Supported edge, but still requires execution discipline."
    if tradability >= 45 and fill_rate >= 0.52:
        return "PROBE", "Some edge support, but not enough for normal size."
    return "DO_NOTHING", "Tradability score too weak."


def build_state(regime: RowDict, openr: RowDict, opt: RowDict, sig: RowDict) -> RowDict:
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


def rank_key(row: RowDict) -> tuple[int, float, int]:
    return (
        to_int(row.get("_match_score"), 0),
        to_float(row.get("tradability_score")) or float("-inf"),
        to_int(row.get("n"), 0),
    )


def choose_matches(
    matrix: list[RowDict], state: RowDict
) -> tuple[RowDict | None, list[RowDict], list[RowDict]]:
    enriched = []
    for row in matrix:
        clone = dict(row)
        clone["_match_score"] = match_score(clone, state)
        enriched.append(clone)

    supported = [row for row in enriched if to_int(row.get("n"), 0) >= MIN_N]
    low_sample = [row for row in enriched if to_int(row.get("n"), 0) < MIN_N]

    ranked = sorted(supported, key=rank_key, reverse=True)
    low_ranked = sorted(low_sample, key=rank_key, reverse=True)
    best = ranked[0] if ranked else None
    return best, ranked[:10], low_ranked[:5]


def risk_label(best: RowDict | None) -> tuple[str, str]:
    if best is None:
        return "NA", "NA"

    failed_fill = to_float(best.get("failed_fill_rate")) or 0.0
    stop_proxy = to_float(best.get("stop_out_rate_proxy")) or 0.0
    max_dd = abs(to_float(best.get("max_drawdown")) or 0.0)

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


def lines_for_match(match: RowDict | None) -> list[str]:
    if match is None:
        return [
            "- Sample support: NONE",
            "- Expected fill probability: NA",
            "- Expected time-to-fill: NA",
            "- Expected MAE before fill: NA",
        ]

    quality = "LOW_SAMPLE" if to_int(match.get("n"), 0) < MIN_N else "SUPPORTED"
    return [
        f"- Sample support: {quality} / n={to_int(match.get('n'), 0)}",
        f"- Expected fill probability: {pct(match.get('fill_rate'))}",
        f"- Direct fill rate: {pct(match.get('direct_fill_rate'))}",
        f"- Failed fill rate: {pct(match.get('failed_fill_rate'))}",
        f"- Expected time-to-fill: {num(match.get('avg_time_to_fill_minutes'))} minutes",
        f"- Median time-to-fill: {num(match.get('median_time_to_fill_minutes'))} minutes",
        f"- Expected MAE before fill: {pct(match.get('avg_MAE_pct'))}",
        f"- Expected MFE: {pct(match.get('avg_MFE_pct'))}",
        f"- Payoff ratio: {num(match.get('payoff_ratio'))}",
        f"- Expectancy: {num(match.get('expectancy'), 4)}",
        f"- Sortino: {num(match.get('sortino_ratio'))}",
        f"- Max drawdown: {pct(match.get('max_drawdown'))}",
        f"- Tradability score: {num(match.get('tradability_score'))}",
    ]


def write_outputs(
    state: RowDict, best: RowDict | None, top: list[RowDict], low: list[RowDict]
) -> None:
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

    md: list[str] = []
    md.append("# SharpEdge 2.0 Auction Expectancy Report")
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
    md.append(
        f"- Open resolution: {state['open_regime_label']} confidence={num(state['regime_confidence'])}"
    )
    md.append(f"- Setup direction: {state['setup_dir']}")
    md.append(f"- Dealer hint: {state['dealer_state_hint']}")
    md.append(f"- Spot: {num(state['spot'])} | ATM: {num(state['atm_strike'])}")
    md.append(f"- Gamma / OI wall: {num(state['gamma_wall'])}")
    md.append(
        f"- Call wall: {num(state['max_call_oi_strike'])} | Put wall: {num(state['max_put_oi_strike'])}"
    )
    md.append(f"- PCR OI: {num(state['pcr_oi'])}")
    md.append(
        f"- Signal bucket: {state['early_bucket']} score={num(state['early_score'])}"
    )
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
    if not top:
        md.append("- No supported historical paths met the sample guard.")
    else:
        for index, row in enumerate(top, 1):
            md.append(
                f"{index}. match={to_int(row.get('_match_score'), 0)} | "
                f"n={to_int(row.get('n'), 0)} | "
                f"event={safe_text(row.get('event_type'))} | "
                f"path={safe_text(row.get('fill_path_type'))} | "
                f"regime={safe_text(row.get('regime_id'))} | "
                f"open={safe_text(row.get('open_regime_label'))} | "
                f"fill={pct(row.get('fill_rate'))} | "
                f"expectancy={num(row.get('expectancy'), 4)} | "
                f"score={num(row.get('tradability_score'))}"
            )

    md.append("")
    md.append("## Low-Sample Rows To Watch But Not Trust")
    if not low:
        md.append("- None.")
    else:
        for index, row in enumerate(low, 1):
            md.append(
                f"{index}. LOW_SAMPLE n={to_int(row.get('n'), 0)} | "
                f"event={safe_text(row.get('event_type'))} | "
                f"path={safe_text(row.get('fill_path_type'))} | "
                f"fill={pct(row.get('fill_rate'))} | "
                f"expectancy={num(row.get('expectancy'), 4)} | "
                f"score={num(row.get('tradability_score'))}"
            )

    card = [
        "SHARPEDGE 2.0 LATEST CARD",
        f"Date: {state['date']} | Symbol: {SYMBOL}",
        f"State: {state['regime_id']} | {state['open_regime_label']} | {state['early_bucket']}",
        f"Recommendation: {recommendation}",
        f"Reason: {rec_reason}",
        f"Fill Prob: {pct(best.get('fill_rate') if best else None)}",
        f"Time-to-fill: {num(best.get('avg_time_to_fill_minutes') if best else None)} min",
        f"MAE before fill: {pct(best.get('avg_MAE_pct') if best else None)}",
        f"Squeeze Risk: {squeeze_risk}",
        f"Continuation Risk: {continuation_risk}",
        f"Playbook: {playbook}",
        f"Sample: n={to_int(best.get('n'), 0) if best else 0} / min={MIN_N}",
    ]

    discord = (
        f"SharpEdge 2.0 | {SYMBOL} {state['date']}\n"
        f"State: {state['regime_id']} | {state['open_regime_label']} | {state['early_bucket']}\n"
        f"Rec: {recommendation} — {rec_reason}\n"
        f"Fill: {pct(best.get('fill_rate') if best else None)} | "
        f"TTF: {num(best.get('avg_time_to_fill_minutes') if best else None)}m | "
        f"MAE: {pct(best.get('avg_MAE_pct') if best else None)}\n"
        f"Squeeze: {squeeze_risk} | Continuation: {continuation_risk}\n"
        f"Sample: n={to_int(best.get('n'), 0) if best else 0}/min={MIN_N}"
    )[:1900]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    CARD_TXT.write_text("\n".join(card) + "\n", encoding="utf-8")
    DISCORD_TXT.write_text(discord + "\n", encoding="utf-8")

    print(f"OK: wrote {REPORT_MD}")
    print(f"OK: wrote {CARD_TXT}")
    print(f"OK: wrote {DISCORD_TXT}")


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        require_tables(con)
        regime = read_latest(con, "regime_daily", "date", "symbol")
        openr = read_latest(con, "open_resolution_regime", "session_date", "underlying")
        opt = read_latest(
            con, "options_positioning_metrics", "snapshot_ts", "underlying"
        )
        sig = read_latest(con, "signals_daily", "date", "symbol")
        state = build_state(regime, openr, opt, sig)
        matrix = load_matrix(con)
        best, top, low = choose_matches(matrix, state)
        write_outputs(state, best, top, low)
    finally:
        con.close()


if __name__ == "__main__":
    main()
