"""Canonical OPEN-resolution classifier for SharpEdge.

SINGLE SOURCE OF TRUTH for classifying how the opening 30 minutes resolved
against yesterday's true key levels (prior_key_low / prior_key_high). The
labeling logic in ``classify()`` is ported verbatim from
``scripts/build_open_resolution_regime_intraday.py`` so the live cockpit read
and the historical backtest table (``open_resolution_regime``) cannot diverge.

Seam: the LABELING (classify) is canonical and identical everywhere. Only the
DATA PREP differs by source — the batch reads 15m bars from SQLite with NY-aware
timestamps; the live cockpit builds equivalent premarket stats + two 15m open
bars from Yahoo 1m bars tagged with exchange-local minute-of-day.

Labels:
  DOWN setup: FAILED_BREAKDOWN_OPEN / ACCEPTED_BREAKDOWN_OPEN_2BAR /
              ACCEPTED_BREAKDOWN_OPEN_STRONG / UNRESOLVED_OPEN
  UP setup:   FAILED_BREAKOUT_OPEN / ACCEPTED_BREAKOUT_OPEN_2BAR /
              ACCEPTED_BREAKOUT_OPEN_STRONG / UNRESOLVED_OPEN
  else:       NO_SETUP / MISSING_RTH
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Canonical thresholds — identical defaults to the batch script.
PM_RETURN_THRESH = float(os.getenv("PM_RETURN_THRESH", "-0.003"))  # -0.30% flush
PM_UP_RETURN_THRESH = float(os.getenv("PM_UP_RETURN_THRESH", "0.003"))  # +0.30% rip
PM_RANGE_RATIO_THRESH = float(os.getenv("PM_RANGE_RATIO_THRESH", "0.0025"))  # 0.25%

# Minute-of-day boundaries in EXCHANGE LOCAL time (America/New_York).
PREMARKET_START_MIN = 4 * 60  # 04:00
RTH_OPEN_MIN = 9 * 60 + 30  # 09:30
RTH_BAR1_END_MIN = 9 * 60 + 45  # 09:45
RTH_BAR2_END_MIN = 10 * 60  # 10:00


# --------------------------- canonical classifier ---------------------------
def classify(
    session_date: str,
    pm: dict[str, float | None],
    keys: dict[str, float | None],
    rths: list[dict[str, Any]],
) -> dict[str, Any]:
    """VERBATIM canonical open-resolution classifier.

    pm: premarket stats dict (pm_open/high/low/close/return/range/range_ratio).
    keys: {prior_key_low, prior_key_high} — today's TRUE key levels.
    rths: up to two 15m opening bars (09:30-09:45, 09:45-10:00).
    """
    notes: list[str] = []
    pm_return = pm["pm_return"]
    pm_rr = pm["pm_range_ratio"]

    has_range = pm_rr is not None and pm_rr >= PM_RANGE_RATIO_THRESH
    flush_down = pm_return is not None and pm_return <= PM_RETURN_THRESH
    rip_up = pm_return is not None and pm_return >= PM_UP_RETURN_THRESH

    setup_dir = "NONE"
    if has_range and flush_down:
        setup_dir = "DOWN"
    elif has_range and rip_up:
        setup_dir = "UP"

    if setup_dir == "NONE":
        return {
            "pm_initiative_flush": 0,
            "setup_dir": "NONE",
            "key_source": None,
            "failed_breakdown_open": 0,
            "accepted_breakdown_open": 0,
            "open_regime_label": "NO_SETUP",
            "regime_confidence": 10.0,
            "notes": "no initiative flush/rip setup",
            "break_level": None,
            "flush_low": pm["pm_low"],
        }

    break_level = None
    key_source = None
    if setup_dir == "DOWN":
        break_level = keys.get("prior_key_low")
        key_source = "PRIOR_KEY_LOW"
    else:
        break_level = keys.get("prior_key_high")
        key_source = "PRIOR_KEY_HIGH"

    if break_level is None:
        break_level = pm["pm_open"]
        key_source = "FALLBACK_PM_OPEN"
        notes.append("break_level fallback=pm_open (true key missing)")

    if not rths:
        return {
            "pm_initiative_flush": 1,
            "setup_dir": setup_dir,
            "key_source": key_source,
            "failed_breakdown_open": 0,
            "accepted_breakdown_open": 0,
            "open_regime_label": "MISSING_RTH",
            "regime_confidence": 0.0,
            "notes": "missing first RTH bar",
            "break_level": break_level,
            "flush_low": pm["pm_low"],
        }

    b1 = rths[0]
    b2 = rths[1] if len(rths) > 1 else None
    o1, h1, l1, c1 = map(float, (b1["open"], b1["high"], b1["low"], b1["close"]))
    c2 = float(b2["close"]) if b2 else None

    flush_low = pm["pm_low"]
    flush_high = pm["pm_high"]

    failed = 0
    accepted = 0
    label = "UNRESOLVED_OPEN"
    conf = 35.0

    if setup_dir == "DOWN":
        swept = l1 < break_level
        reclaimed = swept and (c1 > break_level)
        accept_close_1 = c1 < break_level
        accept_close_2 = c2 is not None and c2 < break_level

        if reclaimed:
            failed = 1
            label = "FAILED_BREAKDOWN_OPEN"
            conf = 72.0
            notes.append("swept below prior_key_low and closed back above")
        elif accept_close_1 and accept_close_2:
            accepted = 1
            label = "ACCEPTED_BREAKDOWN_OPEN_2BAR"
            conf = 70.0
            notes.append("accepted below prior_key_low for 2 bars")
        elif flush_low is not None and c1 < float(flush_low) and (c2 is None or c2 < float(flush_low)):
            accepted = 1
            label = "ACCEPTED_BREAKDOWN_OPEN_STRONG"
            conf = 75.0
            notes.append("accepted below premarket flush low")
        else:
            notes.append("no reclaim/accept yet (down setup)")
    else:  # UP setup
        swept = h1 > break_level
        rejected = swept and (c1 < break_level)
        accept_close_1 = c1 > break_level
        accept_close_2 = c2 is not None and c2 > break_level

        if rejected:
            failed = 1
            label = "FAILED_BREAKOUT_OPEN"
            conf = 72.0
            notes.append("swept above prior_key_high and closed back below")
        elif accept_close_1 and accept_close_2:
            accepted = 1
            label = "ACCEPTED_BREAKOUT_OPEN_2BAR"
            conf = 70.0
            notes.append("accepted above prior_key_high for 2 bars")
        elif flush_high is not None and c1 > float(flush_high) and (c2 is None or c2 > float(flush_high)):
            accepted = 1
            label = "ACCEPTED_BREAKOUT_OPEN_STRONG"
            conf = 75.0
            notes.append("accepted above premarket high")
        else:
            notes.append("no reject/accept yet (up setup)")

    return {
        "pm_initiative_flush": 1,
        "setup_dir": setup_dir,
        "key_source": key_source,
        "failed_breakdown_open": failed,
        "accepted_breakdown_open": accepted,
        "open_regime_label": label,
        "regime_confidence": float(min(conf, 100.0)),
        "notes": "; ".join(notes),
        "break_level": break_level,
        "flush_low": float(flush_low) if flush_low is not None else None,
    }


# --------------------------- live data prep ---------------------------
def _agg_bar(bars: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not bars:
        return None
    return {
        "open": float(bars[0]["open"]),
        "high": float(max(b["high"] for b in bars)),
        "low": float(min(b["low"] for b in bars)),
        "close": float(bars[-1]["close"]),
    }


def premarket_stats_from_minute_bars(day_bars: list[dict[str, Any]]) -> dict[str, float | None]:
    """Build premarket stats from 1m bars tagged with exchange-local minute_of_day."""
    pm = [
        b
        for b in day_bars
        if PREMARKET_START_MIN <= int(b.get("minute_of_day", -1)) < RTH_OPEN_MIN
    ]
    if not pm:
        return {
            "pm_open": None, "pm_high": None, "pm_low": None, "pm_close": None,
            "pm_return": None, "pm_range": None, "pm_range_ratio": None,
        }
    pm_open = float(pm[0]["open"])
    pm_close = float(pm[-1]["close"])
    pm_high = float(max(b["high"] for b in pm))
    pm_low = float(min(b["low"] for b in pm))
    pm_range = pm_high - pm_low
    return {
        "pm_open": pm_open,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "pm_close": pm_close,
        "pm_return": (pm_close / pm_open - 1.0) if pm_open else None,
        "pm_range": pm_range,
        "pm_range_ratio": (pm_range / pm_open) if pm_open else None,
    }


def first_two_open_bars_from_minute_bars(day_bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate 1m RTH bars into the two 15m opening bars the classifier expects."""
    bar1_src = [b for b in day_bars if RTH_OPEN_MIN <= int(b.get("minute_of_day", -1)) < RTH_BAR1_END_MIN]
    bar2_src = [b for b in day_bars if RTH_BAR1_END_MIN <= int(b.get("minute_of_day", -1)) < RTH_BAR2_END_MIN]
    out = []
    b1 = _agg_bar(bar1_src)
    if b1:
        out.append(b1)
        b2 = _agg_bar(bar2_src)
        if b2:
            out.append(b2)
    return out


_PHASE_STORY = {
    "FAILED_BREAKDOWN_OPEN": "Open swept below prior key low and reclaimed — failed breakdown. Long-lean.",
    "ACCEPTED_BREAKDOWN_OPEN_2BAR": "Open accepted below prior key low for two bars — breakdown holding. Short-lean.",
    "ACCEPTED_BREAKDOWN_OPEN_STRONG": "Open accepted below the premarket flush low — strong breakdown. Short-lean.",
    "FAILED_BREAKOUT_OPEN": "Open swept above prior key high and rejected — failed breakout. Short-lean.",
    "ACCEPTED_BREAKOUT_OPEN_2BAR": "Open accepted above prior key high for two bars — breakout holding. Long-lean.",
    "ACCEPTED_BREAKOUT_OPEN_STRONG": "Open accepted above the premarket high — strong breakout. Long-lean.",
    "UNRESOLVED_OPEN": "Initiative setup present but the open has not resolved sweep-fail vs accept yet.",
    "NO_SETUP": "No premarket initiative flush/rip — no open-resolution setup today.",
    "MISSING_RTH": "Premarket setup present; waiting on the first regular-session bars.",
}


def build_open_resolution_live(
    day_bars: list[dict[str, Any]],
    *,
    prior_key_low: float | None,
    prior_key_high: float | None,
    session_date: str = "",
    symbol: str = "SPY",
) -> dict[str, Any]:
    """Compute today's OPEN resolution live from premarket-inclusive 1m bars.

    day_bars: 1m bars tagged with exchange-local minute_of_day (premarket + RTH).
    Returns an ``sharpedge.open_resolution.v1`` packet using the canonical
    classifier, plus a phase flag for pre-open / forming / resolved.
    """
    pm = premarket_stats_from_minute_bars(day_bars)
    keys = {"prior_key_low": prior_key_low, "prior_key_high": prior_key_high}
    rths = first_two_open_bars_from_minute_bars(day_bars)
    cls = classify(session_date, pm, keys, rths)

    label = str(cls["open_regime_label"])
    if label in {"FAILED_BREAKDOWN_OPEN", "FAILED_BREAKOUT_OPEN"}:
        lean = "CALLS" if label == "FAILED_BREAKDOWN_OPEN" else "PUTS"
    elif label.startswith("ACCEPTED_BREAKOUT"):
        lean = "CALLS"
    elif label.startswith("ACCEPTED_BREAKDOWN"):
        lean = "PUTS"
    else:
        lean = "NEUTRAL"

    if label == "NO_SETUP":
        phase = "no_setup"
    elif label in {"MISSING_RTH", "UNRESOLVED_OPEN"}:
        phase = "forming"
    else:
        phase = "resolved"

    return {
        "schema": "sharpedge.open_resolution.v1",
        "symbol": symbol,
        "available": pm["pm_open"] is not None,
        "open_regime_label": label,
        "confidence": int(round(float(cls["regime_confidence"] or 0))),
        "setup_dir": cls["setup_dir"],
        "key_source": cls["key_source"],
        "break_level": cls["break_level"],
        "lean": lean,
        "phase": phase,
        "story": _PHASE_STORY.get(label, label),
        "premarket": {
            "pm_return": pm["pm_return"],
            "pm_range_ratio": pm["pm_range_ratio"],
            "pm_low": pm["pm_low"],
            "pm_high": pm["pm_high"],
        },
        "notes": cls["notes"],
        "source": "live:yahoo_1m_prepost",
        "classifier": "canonical:open_resolution.classify",
    }


__all__ = [
    "PM_RANGE_RATIO_THRESH",
    "PM_RETURN_THRESH",
    "PM_UP_RETURN_THRESH",
    "build_open_resolution_live",
    "classify",
    "first_two_open_bars_from_minute_bars",
    "premarket_stats_from_minute_bars",
]
