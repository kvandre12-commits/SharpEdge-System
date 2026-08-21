from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from event_calendar import EARNINGS_DATES
from market_data_sources import (
    fetch_yahoo_daily_bars,
    fetch_yahoo_regular_session_chart_rows,
)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
STUDY_PATH = (
    OUTPUTS_DIR
    / "apple_earnings_reaction_dips_top8"
    / "apple_earnings_reaction_dips.json"
)
OPTION_BOARD_CANDIDATES = (
    OUTPUTS_DIR / "nerv_watchlist" / "nerv_liquidity_board.json",
    OUTPUTS_DIR / "nerv_cockpit_standard" / "nerv_liquidity_board.json",
    OUTPUTS_DIR / "nerv" / "nerv_liquidity_board.json",
)
BENCHMARK_SYMBOLS = {"SPY", "QQQ"}
WINDOW_SESSIONS = 5

SessionFetcher = Callable[[str], tuple[list[dict[str, Any]], dict[str, Any]]]
DailyFetcher = Callable[[str], tuple[list[dict[str, Any]], dict[str, Any]]]


def _default_daily_fetcher(symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return fetch_yahoo_daily_bars(symbol, range_="6mo")


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _pct_change(current: float | None, reference: float | None) -> float | None:
    if current is None or reference in (None, 0):
        return None
    return ((float(current) / float(reference)) - 1.0) * 100.0


def _load_study(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _session_snapshot(
    symbol: str,
    *,
    fetcher: SessionFetcher,
) -> dict[str, Any]:
    rows, source = fetcher(symbol)
    if not rows:
        return {
            "symbol": symbol,
            "available": False,
            "reason": "no intraday rows returned",
            "source": source,
        }

    previous_close = source.get("chart_previous_close")
    open_price = rows[0].get("open")
    last_price = rows[-1].get("close")
    high_price = max(float(row["high"]) for row in rows)
    low_price = min(float(row["low"]) for row in rows)
    total_volume = sum(int(row.get("volume") or 0) for row in rows)
    day_change_pct = _pct_change(last_price, previous_close)
    open_gap_pct = _pct_change(open_price, previous_close)
    range_position_pct = None
    if high_price > low_price and last_price is not None:
        range_position_pct = (
            (float(last_price) - low_price) / (high_price - low_price)
        ) * 100.0

    return {
        "symbol": symbol,
        "available": True,
        "session_date": rows[-1].get("date"),
        "open": _round(open_price, 4),
        "last": _round(last_price, 4),
        "high": _round(high_price, 4),
        "low": _round(low_price, 4),
        "previous_close": _round(previous_close, 4),
        "day_change_pct": _round(day_change_pct),
        "open_gap_pct": _round(open_gap_pct),
        "range_position_pct": _round(range_position_pct, 1),
        "bar_count": len(rows),
        "volume": total_volume,
        "source": source,
    }


def _relative_pct_points(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return _round(value - base)


def _latest_verified_earnings_date(
    session_date: str | None,
    earnings_dates: list[str],
) -> str | None:
    if not session_date:
        return None
    eligible = [date for date in earnings_dates if date <= session_date]
    return max(eligible) if eligible else None


def _resolve_verified_window(
    *,
    session_date: str | None,
    earnings_dates: list[str],
    daily_fetcher: DailyFetcher,
) -> dict[str, Any]:
    if not session_date:
        return {
            "available": False,
            "active": False,
            "reason": "missing session date anchor",
        }

    earnings_date = _latest_verified_earnings_date(session_date, earnings_dates)
    if not earnings_date:
        return {
            "available": False,
            "active": False,
            "reason": "no verified AAPL earnings date before this session",
        }

    daily_rows, daily_source = daily_fetcher("AAPL")
    index_by_date = {str(row.get("date")): idx for idx, row in enumerate(daily_rows)}
    earnings_idx = index_by_date.get(earnings_date)
    session_idx = index_by_date.get(session_date)
    if earnings_idx is None:
        return {
            "available": False,
            "active": False,
            "reason": f"verified earnings date {earnings_date} missing from AAPL daily bars",
            "source": daily_source,
        }
    if session_idx is None:
        return {
            "available": False,
            "active": False,
            "reason": f"session date {session_date} missing from AAPL daily bars",
            "source": daily_source,
        }
    if earnings_idx + 1 >= len(daily_rows):
        return {
            "available": False,
            "active": False,
            "reason": "reaction day not yet available in AAPL daily bars",
            "source": daily_source,
        }

    earnings_row = daily_rows[earnings_idx]
    reaction_row = daily_rows[earnings_idx + 1]
    reaction_session_date = str(reaction_row.get("date"))
    sessions_since_reaction = session_idx - (earnings_idx + 1)
    in_window = 0 <= sessions_since_reaction <= WINDOW_SESSIONS
    reaction_open_gap_pct = _pct_change(
        reaction_row.get("open"),
        earnings_row.get("close"),
    )
    reaction_close_vs_prior_close_pct = _pct_change(
        reaction_row.get("close"),
        earnings_row.get("close"),
    )
    active = bool(
        in_window and reaction_open_gap_pct is not None and reaction_open_gap_pct < 0
    )

    phase_label = "outside_window"
    if sessions_since_reaction == 0:
        phase_label = "reaction_day"
    elif sessions_since_reaction == 1:
        phase_label = "day_2_followthrough"
    elif 2 <= sessions_since_reaction <= WINDOW_SESSIONS:
        phase_label = "post_dip_followthrough"

    if active:
        reason = (
            f"verified AAPL AMC earnings {earnings_date} -> reaction {reaction_session_date} "
            f"opened {reaction_open_gap_pct:.1f}% below prior close; current session is day {sessions_since_reaction + 1} of the post-dip window"
        )
    elif in_window:
        reason = (
            f"verified AAPL earnings window is open, but the reaction day did not lower-open "
            f"({reaction_open_gap_pct:.1f}% gap)"
            if reaction_open_gap_pct is not None
            else "verified AAPL earnings window is open, but reaction-day gap could not be confirmed"
        )
    else:
        reason = (
            f"current session is {sessions_since_reaction} sessions from the AAPL reaction day; "
            f"outside the {WINDOW_SESSIONS}-session trade window"
        )

    return {
        "available": True,
        "active": active,
        "in_window": in_window,
        "phase_label": phase_label,
        "earnings_date": earnings_date,
        "reaction_session_date": reaction_session_date,
        "current_session_date": session_date,
        "sessions_since_reaction": sessions_since_reaction,
        "reaction_open_gap_pct": _round(reaction_open_gap_pct),
        "reaction_close_vs_prior_close_pct": _round(reaction_close_vs_prior_close_pct),
        "reason": reason,
        "source": daily_source,
    }


def _trade_lane(row: dict[str, Any]) -> dict[str, Any]:
    rel_qqq = row.get("relative_to_qqq_pct_points")
    day_change = row.get("live_day_change_pct")
    open_gap = row.get("live_open_gap_pct")
    range_pos = row.get("live_range_position_pct")

    if row.get("role") == "benchmark":
        return {
            "actionable_today": False,
            "execution_bias": "NEUTRAL",
            "lane": "benchmark_only",
            "lane_label": "BENCHMARK ONLY",
            "reason": "Use SPY/QQQ for tape context, not as the post-AAPL single-name trade candidate.",
            "stretch_risk": False,
        }

    if rel_qqq is None or day_change is None or range_pos is None:
        return {
            "actionable_today": False,
            "execution_bias": "NEUTRAL",
            "lane": "no_data",
            "lane_label": "NO DATA",
            "reason": "Missing live relative-strength inputs.",
            "stretch_risk": False,
        }

    strong_leader = rel_qqq >= 1.0 and day_change > 0 and range_pos >= 65
    reclaim_candidate = rel_qqq >= 0.4 and day_change > -0.5 and range_pos >= 45
    stretched = bool((open_gap or 0.0) >= 5.0 and day_change >= 5.0)

    if strong_leader:
        if stretched:
            return {
                "actionable_today": True,
                "execution_bias": "CALLS",
                "lane": "pullback_reclaim_long",
                "lane_label": "LONG ONLY ON PULLBACK / RECLAIM",
                "reason": "Real relative-strength leader, but already extended. Do not chase the face-ripper candle; buy the pullback if it reclaims.",
                "stretch_risk": True,
            }
        return {
            "actionable_today": True,
            "execution_bias": "CALLS",
            "lane": "trend_continuation_long",
            "lane_label": "GO WITH STRENGTH",
            "reason": "Outperforming QQQ/SPY and still living in the upper session range. This is strength, not cope.",
            "stretch_risk": False,
        }

    if reclaim_candidate:
        return {
            "actionable_today": True,
            "execution_bias": "CALLS",
            "lane": "reclaim_long",
            "lane_label": "BUY DIP ONLY IF RECLAIM HOLDS",
            "reason": "Still leading enough to matter, but not clean enough to chase. Needs intraday reclaim/acceptance, not prayer.",
            "stretch_risk": False,
        }

    weak_note = (
        "flat-to-weak vs QQQ" if (rel_qqq or 0.0) < 0.4 else "low range acceptance"
    )
    return {
        "actionable_today": False,
        "execution_bias": "NEUTRAL",
        "lane": "no_trade",
        "lane_label": "NO TRADE",
        "reason": f"Not a live leader today ({weak_note}). If it can't lead the tape, it doesn't get the money.",
        "stretch_risk": False,
    }


def _rank_score(row: dict[str, Any], *, window_active: bool) -> float:
    historical_edge = float(row.get("historical_median_5d_return_pct") or 0.0)
    rel_qqq = float(row.get("relative_to_qqq_pct_points") or 0.0)
    rel_spy = float(row.get("relative_to_spy_pct_points") or 0.0)
    day_change = float(row.get("live_day_change_pct") or 0.0)
    range_pos = float(row.get("live_range_position_pct") or 0.0)
    lane_bonus = 4.0 if row.get("actionable_today") else -4.0
    stretch_penalty = -2.0 if row.get("stretch_risk") else 0.0
    if window_active:
        return round(
            (rel_qqq * 2.4)
            + (rel_spy * 0.8)
            + (day_change * 0.35)
            + ((range_pos - 50.0) * 0.08)
            + (historical_edge * 0.2)
            + lane_bonus
            + stretch_penalty,
            4,
        )
    return round((historical_edge * 0.4) + lane_bonus, 4)


def _preferred_option_type(execution_bias: str | None) -> str:
    return "call" if str(execution_bias or "").upper() == "CALLS" else ""


def _option_contract_sort_key(
    contract: dict[str, Any],
    *,
    preferred_type: str,
) -> tuple[float, ...]:
    option_type = str(contract.get("option_type") or "").lower()
    type_bonus = 1.0 if preferred_type and option_type == preferred_type else 0.0
    priority = str(contract.get("manual_validation_priority") or "")
    priority_score = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(priority, 0.0)
    nerv_score = float(contract.get("nerv_score") or 0.0)
    liquidity_score = float(contract.get("liquidity_score") or 0.0)
    volume = float(contract.get("volume") or 0.0)
    open_interest = float(contract.get("open_interest") or 0.0)
    width_pct = float(contract.get("width_pct") or 999.0)
    return (
        type_bonus,
        priority_score,
        nerv_score,
        liquidity_score,
        volume,
        open_interest,
        -width_pct,
    )


def _load_option_liquidity(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    requested_symbols = {
        str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")
    }
    best_board_path: Path | None = None
    best_payload: dict[str, Any] | None = None
    best_contracts: list[dict[str, Any]] = []
    best_score = (-1, -1.0)
    read_errors: list[str] = []

    for candidate in OPTION_BOARD_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            read_errors.append(f"{candidate}: {exc}")
            continue
        contracts = payload.get("contracts") or payload.get("rows") or []
        covered_symbols = {
            str(contract.get("underlying") or contract.get("symbol") or "").upper()
            for contract in contracts
        }
        coverage_count = len(requested_symbols & covered_symbols)
        score = (coverage_count, candidate.stat().st_mtime)
        if score > best_score:
            best_score = score
            best_board_path = candidate
            best_payload = payload
            best_contracts = contracts

    if best_board_path is None or best_payload is None:
        reason = "no option liquidity board found"
        if read_errors:
            reason = f"option liquidity boards unreadable: {'; '.join(read_errors)}"
        return {}, {"available": False, "reason": reason}

    board_path = best_board_path
    payload = best_payload
    contracts = best_contracts
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for contract in contracts:
        symbol = str(contract.get("underlying") or contract.get("symbol") or "").upper()
        if symbol:
            by_symbol.setdefault(symbol, []).append(contract)

    best_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        candidates = by_symbol.get(symbol) or []
        if not candidates:
            continue
        preferred_type = _preferred_option_type(row.get("execution_bias"))
        best = max(
            candidates,
            key=lambda contract: _option_contract_sort_key(
                contract,
                preferred_type=preferred_type,
            ),
        )
        best_by_symbol[symbol] = {
            "available": True,
            "symbol": symbol,
            "expiration": best.get("expiration"),
            "option_type": best.get("option_type"),
            "strike": _round(best.get("strike"), 2),
            "bid": _round(best.get("bid"), 2),
            "ask": _round(best.get("ask"), 2),
            "midpoint": _round(best.get("midpoint"), 2),
            "width_pct": _round(
                (float(best.get("width_pct")) * 100.0)
                if best.get("width_pct") is not None
                else None,
                1,
            ),
            "volume": int(float(best.get("volume") or 0)),
            "open_interest": int(float(best.get("open_interest") or 0)),
            "quote_age_seconds": int(float(best.get("quote_age_seconds") or 0)),
            "fresh_quote_required": bool(best.get("fresh_quote_required")),
            "contract_symbol": best.get("contract_symbol"),
            "priority": best.get("manual_validation_priority"),
            "score": _round(best.get("nerv_score"), 2),
            "source": best.get("source"),
        }

    source = {
        "available": True,
        "path": str(board_path),
        "updated_at_utc": max(
            (
                contract.get("fetch_timestamp")
                for contract in contracts
                if contract.get("fetch_timestamp")
            ),
            default=None,
        ),
    }
    return best_by_symbol, source


def build_post_apple_rotation_live(
    *,
    study_path: Path | None = None,
    fetcher: SessionFetcher = fetch_yahoo_regular_session_chart_rows,
    daily_fetcher: DailyFetcher = _default_daily_fetcher,
    earnings_dates: list[str] | None = None,
) -> dict[str, Any]:
    path = study_path or STUDY_PATH
    if not path.exists():
        return {
            "schema": "sharpedge.post_apple_rotation.v1",
            "available": False,
            "reason": f"study file missing: {path}",
            "source": "study:apple_earnings_reaction_dips",
        }

    study = _load_study(path)
    summaries = study.get("summaries") or {}
    symbols = [symbol for symbol in study.get("symbols") or [] if symbol in summaries]
    if "AAPL" not in symbols:
        return {
            "schema": "sharpedge.post_apple_rotation.v1",
            "available": False,
            "reason": "study payload missing AAPL summary",
            "source": str(path),
        }

    snapshots = {
        symbol: _session_snapshot(symbol, fetcher=fetcher) for symbol in symbols
    }
    aapl = snapshots.get("AAPL") or {}
    qqq = snapshots.get("QQQ") or {}
    spy = snapshots.get("SPY") or {}
    session_date = (
        aapl.get("session_date") or qqq.get("session_date") or spy.get("session_date")
    )
    verified_window = _resolve_verified_window(
        session_date=session_date,
        earnings_dates=earnings_dates or EARNINGS_DATES.get("AAPL", []),
        daily_fetcher=daily_fetcher,
    )

    leaderboard: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol == "AAPL":
            continue
        conditional = (summaries.get(symbol) or {}).get(
            "reaction_opened_below_prior_close"
        ) or {}
        snapshot = snapshots.get(symbol) or {}
        row = {
            "symbol": symbol,
            "role": "benchmark" if symbol in BENCHMARK_SYMBOLS else "leader",
            "historical_sample_size": conditional.get("count"),
            "historical_median_5d_return_pct": _round(
                conditional.get("median_return_5d_pct")
            ),
            "historical_5d_positive_pct": _round(
                conditional.get("return_5d_positive_pct")
            ),
            "historical_median_3d_return_pct": _round(
                conditional.get("median_return_3d_pct")
            ),
            "historical_lower_open_streak": _round(
                conditional.get("median_consecutive_lower_opens_from_reaction"),
                1,
            ),
            "live_available": snapshot.get("available", False),
            "live_day_change_pct": snapshot.get("day_change_pct"),
            "live_open_gap_pct": snapshot.get("open_gap_pct"),
            "live_range_position_pct": snapshot.get("range_position_pct"),
            "relative_to_aapl_pct_points": _relative_pct_points(
                snapshot.get("day_change_pct"),
                aapl.get("day_change_pct"),
            ),
            "relative_to_qqq_pct_points": _relative_pct_points(
                snapshot.get("day_change_pct"),
                qqq.get("day_change_pct"),
            ),
            "relative_to_spy_pct_points": _relative_pct_points(
                snapshot.get("day_change_pct"),
                spy.get("day_change_pct"),
            ),
            "last": snapshot.get("last"),
        }
        row.update(_trade_lane(row))
        row["rank_score"] = _rank_score(
            row,
            window_active=bool(verified_window.get("active")),
        )
        leaderboard.append(row)

    leaderboard.sort(
        key=lambda row: (
            row["rank_score"],
            1 if row.get("actionable_today") else 0,
            row.get("historical_median_5d_return_pct")
            if row.get("historical_median_5d_return_pct") is not None
            else float("-inf"),
        ),
        reverse=True,
    )
    for index, row in enumerate(leaderboard, start=1):
        row["rank"] = index

    option_liquidity_by_symbol, option_liquidity_source = _load_option_liquidity(
        leaderboard
    )
    for row in leaderboard:
        row["options_liquidity"] = option_liquidity_by_symbol.get(
            str(row.get("symbol") or "").upper(),
            {"available": False},
        )

    leaders_only = [row for row in leaderboard if row.get("role") == "leader"]
    trade_candidates = (
        [row for row in leaders_only if row.get("actionable_today")][:3]
        if verified_window.get("active")
        else []
    )
    benchmark_context = [row for row in leaderboard if row.get("role") == "benchmark"]

    if verified_window.get("active") and trade_candidates:
        mode = "trade_today"
        top_symbols = ", ".join(row["symbol"] for row in trade_candidates)
        headline = f"Verified post-AAPL dip day {int(verified_window['sessions_since_reaction']) + 1}: trade {top_symbols}"
        story = (
            "Verified AAPL lower-open reaction window is active. These names are the ones actually leading today, "
            "so this card is trading the tape in front of us, not the fantasy in our head."
        )
    elif verified_window.get("active"):
        mode = "stand_down_context_only"
        headline = f"Verified post-AAPL dip day {int(verified_window['sessions_since_reaction']) + 1}: no clean leader today"
        story = (
            "The verified post-dip window is active, but today's cross-name tape is not giving a clean leader worth forcing. "
            "Stand down until someone actually earns the leash."
        )
    else:
        mode = "inactive_window"
        headline = "No verified post-AAPL lower-open trade window today"
        story = verified_window.get("reason") or (
            "Without a verified reaction-day lower open inside the active window, this card stays context-only."
        )

    return {
        "schema": "sharpedge.post_apple_rotation.v1",
        "available": True,
        "mode": mode,
        "headline": headline,
        "story": story,
        "study_source": {
            "path": str(path),
            "generated_at_utc": study.get("generated_at_utc"),
            "assumption": study.get("assumption"),
        },
        "verified_window": verified_window,
        "ranking_method": (
            "Only activate trading inside the verified AAPL lower-open reaction window. "
            "Then rank leaders by live relative strength, day change, range control, and trade-lane validity; historical study is only a tiebreaker."
        ),
        "benchmarks": {
            "AAPL": aapl,
            "QQQ": qqq,
            "SPY": spy,
        },
        "benchmark_context": benchmark_context,
        "today_trades": trade_candidates,
        "leaderboard": leaderboard,
        "options_liquidity_source": option_liquidity_source,
    }
