"""Live GAP FILL EDGE read for the cockpit.

Surfaces the canonical auction-expectancy history (auction_expectancy_events)
as an actionable live edge: given TODAY's gap direction + the current regime,
how often did comparable days historically fill the gap, how long did it take,
and what was the realized reward/risk?

Why we read the EVENT table, not conditional_expectancy_matrix:
    The matrix groups by outcome-only dimensions (fill_path_type, setup_dir,
    key_source, trade_gate, pressure_state) that are unknowable at the open.
    Matching a live day against those would be look-ahead. Instead we aggregate
    the raw events over ONLY pre-known conditions (gap_direction + regime
    states + open_regime_label), relaxing progressively until we have enough
    sample. That keeps the surfaced edge honest and causal.
"""

from __future__ import annotations

import os
import sqlite3
import statistics
from collections import Counter
from typing import Any, Optional


def _default_db_path() -> str:
    return os.getenv("SPY_DB_PATH") or os.path.expanduser(
        "~/SharpEdge-System/data/spy_truth.db"
    )


def _existing_cols(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _gap_from_open(
    prior_close: Optional[float], today_open: Optional[float]
) -> dict[str, Any]:
    if not prior_close or today_open is None:
        return {"gap_direction": None, "gap_pct": None}
    gap_pct = (today_open - prior_close) / prior_close
    if today_open > prior_close:
        direction = "UP"
    elif today_open < prior_close:
        direction = "DOWN"
    else:
        direction = "FLAT"
    return {"gap_direction": direction, "gap_pct": gap_pct}


def gap_from_daily_bars(daily_bars: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive today's live gap from the cockpit's daily bar list.

    daily_bars are dicts with date/open/high/low/close (last = today's session).
    """
    if not daily_bars or len(daily_bars) < 2:
        return {"gap_direction": None, "gap_pct": None, "session_date": None}
    today = daily_bars[-1]
    prior = daily_bars[-2]
    try:
        info = _gap_from_open(float(prior["close"]), float(today["open"]))
    except (KeyError, TypeError, ValueError):
        return {"gap_direction": None, "gap_pct": None, "session_date": None}
    info["session_date"] = today.get("date")
    return info


def _median_safe(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _mean_safe(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value != value or abs(value) == float("inf"):
        return lo
    return max(lo, min(hi, float(value)))


def _sortino_safe(values: list[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    downside = [v for v in vals if v < 0]
    if len(downside) < 2:
        return None
    mean = sum(vals) / len(vals)
    dmean = sum(downside) / len(downside)
    variance = sum((v - dmean) ** 2 for v in downside) / (len(downside) - 1)
    dstd = variance**0.5
    if dstd <= 0:
        return None
    return float(mean / dstd)


def _timing_quality_components(
    *,
    median_ttf: Optional[float],
    median_mae: Optional[float],
    median_mfe: Optional[float],
    sortino_ratio: Optional[float],
) -> dict[str, float]:
    ttf = float(median_ttf) if median_ttf is not None else 120.0
    if ttf <= 0:
        ttf = 120.0
    mae = abs(float(median_mae or 0.0))
    mfe = abs(float(median_mfe or 0.0))
    sortino = float(sortino_ratio or 0.0)

    if ttf <= 45.0:
        fast_bonus = _clamp((45.0 - ttf) / 45.0)
        slow_penalty = 0.0
    else:
        fast_bonus = 0.0
        slow_penalty = _clamp((ttf - 120.0) / 120.0)

    mae_penalty = _clamp(mae / 0.01)
    asymmetry = _clamp((mfe / max(mae, 0.0001)) / 3.0)
    sortino_quality = _clamp((sortino - -1.0) / 4.0)
    timing_quality = _clamp(
        0.25 * fast_bonus
        - 0.20 * slow_penalty
        + 0.15 * (1.0 - mae_penalty)
        + 0.30 * asymmetry
        + 0.30 * sortino_quality
    )
    return {
        "fast_fill_bonus": fast_bonus,
        "slow_fill_penalty": slow_penalty,
        "median_MAE_penalty": mae_penalty,
        "MFE_asymmetry_score": asymmetry,
        "sortino_quality_score": sortino_quality,
        "timing_quality_score": timing_quality,
    }


def _timing_quality_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.70:
        return "clean"
    if score >= 0.45:
        return "mixed"
    return "choppy/slow"


def build_gap_fill_edge_live(
    db_path: Optional[str] = None,
    *,
    symbol: str = "SPY",
    gap_direction: Optional[str] = None,
    gap_pct: Optional[float] = None,
    vol_state: Optional[str] = None,
    macro_state: Optional[str] = None,
    dp_state: Optional[str] = None,
    open_regime_label: Optional[str] = None,
    session_date: Optional[str] = None,
    min_n: int = 8,
) -> dict[str, Any]:
    """Aggregate historical gap-fill outcomes matching today's causal context."""
    schema = "sharpedge.gap_fill_edge.v1"
    db_path = db_path or _default_db_path()

    if not gap_direction or gap_direction == "FLAT":
        return {
            "schema": schema,
            "available": False,
            "reason": "no directional gap today (flat open)",
            "gap_direction": gap_direction,
            "gap_pct": gap_pct,
            "source": "db:auction_expectancy_events",
        }

    if not os.path.exists(db_path):
        return {
            "schema": schema,
            "available": False,
            "reason": f"expectancy DB not found at {db_path}",
            "source": "db:auction_expectancy_events",
        }

    try:
        con = sqlite3.connect(db_path)
        try:
            tbls = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "auction_expectancy_events" not in tbls:
                return {
                    "schema": schema,
                    "available": False,
                    "reason": "auction_expectancy_events not built yet "
                    "(run scripts/build_auction_expectancy_pipeline.py)",
                    "source": "db:auction_expectancy_events",
                }
            cols = _existing_cols(con, "auction_expectancy_events")
            has_trade_r = "realized_trade_R" in cols
            has_stop_before_fill = "stop_before_fill" in cols
            has_rr = "reward_risk_realized" in cols
            has_mfe = "MFE_pct" in cols
            has_mae = "MAE_pct" in cols
            has_causal_mfe = "mfe_before_resolution_pct" in cols
            has_causal_mae = "mae_before_resolution_pct" in cols
            has_ttf = "time_to_fill_minutes" in cols
            has_path = "fill_path_type" in cols
            has_intraday_flag = "has_intraday_bars" in cols

            select_cols = [
                "fill_completed",
                "vol_state",
                "macro_state",
                "dp_state",
                "open_regime_label",
            ]
            if has_path:
                select_cols.append("fill_path_type")
            if has_ttf:
                select_cols.append("time_to_fill_minutes")
            if has_trade_r:
                select_cols.append("realized_trade_R")
            if has_stop_before_fill:
                select_cols.append("stop_before_fill")
            if has_rr:
                select_cols.append("reward_risk_realized")
            if has_causal_mfe:
                select_cols.append("mfe_before_resolution_pct")
            if has_causal_mae:
                select_cols.append("mae_before_resolution_pct")
            if has_mfe:
                select_cols.append("MFE_pct")
            if has_mae:
                select_cols.append("MAE_pct")

            con.row_factory = sqlite3.Row
            coverage_filter = (
                " AND COALESCE(has_intraday_bars, 0) = 1" if has_intraday_flag else ""
            )
            rows = con.execute(
                f"SELECT {','.join(select_cols)} FROM auction_expectancy_events "
                f"WHERE symbol = ? AND gap_direction = ?{coverage_filter}",
                (symbol, gap_direction),
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:  # never let a bad DB break the cockpit
        return {
            "schema": schema,
            "available": False,
            "reason": f"gap fill edge read failed: {exc}",
            "source": "db:auction_expectancy_events",
        }

    if not rows:
        return {
            "schema": schema,
            "available": False,
            "reason": f"no historical {gap_direction} gaps for {symbol}",
            "gap_direction": gap_direction,
            "gap_pct": gap_pct,
            "source": "db:auction_expectancy_events",
        }

    # Progressive relaxation: most specific causal match first.
    conds = {
        "vol_state": vol_state,
        "macro_state": macro_state,
        "dp_state": dp_state,
        "open_regime_label": open_regime_label,
    }
    levels = [
        ["vol_state", "macro_state", "dp_state", "open_regime_label"],
        ["vol_state", "macro_state", "dp_state"],
        ["vol_state", "macro_state"],
        ["vol_state"],
        [],  # gap_direction only
    ]

    def _match(row: sqlite3.Row, keys: list[str]) -> bool:
        for k in keys:
            want = conds.get(k)
            if not want:
                return False  # can't match on a dimension we don't know live
            if str(row[k]) != str(want):
                return False
        return True

    chosen: list[sqlite3.Row] = []
    matched_on: list[str] = []
    for keys in levels:
        usable = [k for k in keys if conds.get(k)]
        if len(usable) != len(keys):
            continue  # skip levels needing a dim we don't have
        subset = [r for r in rows if _match(r, keys)]
        if len(subset) >= min_n:
            chosen = subset
            matched_on = keys
            break
    if not chosen:
        chosen = rows  # fall back to all same-direction gaps
        matched_on = []

    n = len(chosen)
    fill_rate = _mean_safe([float(r["fill_completed"] or 0) for r in chosen]) or 0.0

    path_mix: list[dict[str, Any]] = []
    path_mix_text = ""
    if has_path:
        counts = Counter(str(r["fill_path_type"] or "UNKNOWN") for r in chosen)
        path_mix = [
            {"path": path, "count": count, "pct": count / n if n else 0.0}
            for path, count in counts.most_common(5)
        ]
        path_mix_text = " · ".join(
            f"{item['path']} {item['pct'] * 100:.0f}%" for item in path_mix[:4]
        )

    ttf_vals: list[float] = []
    if has_ttf:
        for r in chosen:
            v = r["time_to_fill_minutes"]
            if v is not None and int(r["fill_completed"] or 0) == 1:
                try:
                    ttf_vals.append(float(v))
                except (TypeError, ValueError):
                    pass
    median_ttf = _median_safe(ttf_vals)

    stop_before_fill_rate = None
    if has_stop_before_fill:
        stop_before_fill_rate = _mean_safe(
            [float(r["stop_before_fill"] or 0) for r in chosen]
        )

    def _finite_values(column: str) -> list[float]:
        vals = []
        for r in chosen:
            v = r[column]
            if v is None:
                continue
            try:
                fv = float(v)
                if fv == fv and abs(fv) != float("inf"):  # drop NaN/inf
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
        return vals

    expectancy = None
    expectancy_kind = None
    if has_trade_r:
        trade_r = _finite_values("realized_trade_R")
        if trade_r:
            expectancy = _mean_safe(trade_r)
            expectancy_kind = "causal_trade_R"
    if expectancy is None and has_rr:
        rr = _finite_values("reward_risk_realized")
        if rr:
            expectancy = _mean_safe(rr)
            expectancy_kind = "full_session_reward_risk"
    mfe_col = "mfe_before_resolution_pct" if has_causal_mfe else "MFE_pct"
    mae_col = "mae_before_resolution_pct" if has_causal_mae else "MAE_pct"
    median_mae = (
        _median_safe(_finite_values(mae_col)) if (has_causal_mae or has_mae) else None
    )
    median_mfe = (
        _median_safe(_finite_values(mfe_col)) if (has_causal_mfe or has_mfe) else None
    )
    sortino_ratio = (
        _sortino_safe(_finite_values("realized_trade_R")) if has_trade_r else None
    )
    timing_quality = _timing_quality_components(
        median_ttf=median_ttf,
        median_mae=median_mae,
        median_mfe=median_mfe,
        sortino_ratio=sortino_ratio,
    )

    if (
        expectancy is None
        and (has_causal_mfe or has_mfe)
        and (has_causal_mae or has_mae)
    ):
        edge = []
        for r in chosen:
            try:
                mfe = float(r[mfe_col]) if r[mfe_col] is not None else 0.0
                mae = float(r[mae_col]) if r[mae_col] is not None else 0.0
                edge.append(mfe - abs(mae))
            except (TypeError, ValueError):
                pass
        if edge:
            expectancy = _mean_safe(edge)
            expectancy_kind = "causal_mfe_minus_mae_pct"

    sample_quality = (
        "GOOD"
        if n >= max(min_n, 20)
        else ("BOOTSTRAP_OK" if n >= min_n else "LOW_SAMPLE")
    )

    gap_txt = f"{gap_pct * 100:+.2f}%" if isinstance(gap_pct, (int, float)) else ""
    match_txt = (
        "matched on " + ", ".join(matched_on)
        if matched_on
        else f"all historical {gap_direction} gaps"
    )
    exp_txt = ""
    if expectancy is not None:
        if expectancy_kind == "causal_trade_R":
            exp_txt = f" Causal trade expectancy {expectancy:+.2f}R."
        elif expectancy_kind == "full_session_reward_risk":
            exp_txt = f" Full-session reward/risk avg {expectancy:.2f}."
        else:
            exp_txt = f" Avg net excursion {expectancy * 100:+.2f}%."
    story = (
        f"{gap_direction} gap {gap_txt}: historically filled "
        f"{fill_rate * 100:.0f}% of the time (n={n}, {match_txt})."
    )
    if median_ttf is not None:
        story += f" Median time-to-fill {median_ttf:.0f} min."
    if stop_before_fill_rate is not None:
        story += f" Stop-before-fill {stop_before_fill_rate * 100:.0f}%."
    story += exp_txt
    tq = timing_quality["timing_quality_score"]
    story += f" Timing quality {_timing_quality_label(tq)} ({tq * 100:.0f}/100)."
    if sample_quality == "LOW_SAMPLE":
        story += " [low sample — treat as weak prior]"

    return {
        "schema": schema,
        "available": True,
        "symbol": symbol,
        "session_date": session_date,
        "gap_direction": gap_direction,
        "gap_pct": gap_pct,
        "n": n,
        "fill_rate": fill_rate,
        "median_time_to_fill_minutes": median_ttf,
        "expectancy": expectancy,
        "expectancy_kind": expectancy_kind,
        "stop_before_fill_rate": stop_before_fill_rate,
        "median_MAE_pct": median_mae,
        "median_MFE_pct": median_mfe,
        "sortino_ratio": sortino_ratio,
        "timing_quality": timing_quality,
        "path_mix": path_mix,
        "path_mix_text": path_mix_text,
        "matched_on": matched_on,
        "matched_conditions": {k: conds[k] for k in matched_on},
        "sample_quality": sample_quality,
        "headline": f"GAP FILL {fill_rate * 100:.0f}% · {gap_direction} {gap_txt}",
        "story": story,
        "source": "db:auction_expectancy_events",
    }


__all__ = ["build_gap_fill_edge_live", "gap_from_daily_bars"]
