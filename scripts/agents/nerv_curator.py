#!/usr/bin/env python3
"""Curate NERV + IV/RV13 context into an operator-facing read.

This is a deterministic operator agent. It does not authorize trades, route orders,
or replace the approval_decision object. It turns noisy NERV tables into a concise
"what am I looking at?" packet for the human operator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.nerv_curator_language import build_hey_guy_summary
from scripts.agents.nerv_curator_output import (
    render_packet_text,
    write_packet_files,
)

OUTDIR = Path("outputs")
DEFAULT_BOARD_CANDIDATES = (
    OUTDIR / "nerv_cockpit_standard" / "nerv_liquidity_board.json",
    OUTDIR / "nerv_watchlist" / "nerv_liquidity_board.json",
    OUTDIR / "nerv_spy_month" / "nerv_liquidity_board.json",
    OUTDIR / "nerv_spy" / "nerv_liquidity_board.json",
    OUTDIR / "nerv" / "nerv_liquidity_board.json",
)
DEFAULT_IV_HEAT = OUTDIR / "iv_heat_harvest" / "spy_iv_heat_harvest.json"
DEFAULT_SIGNAL = OUTDIR / "signal.json"
DEFAULT_JSON = OUTDIR / "nerv_curator.json"
DEFAULT_TXT = OUTDIR / "nerv_curator.txt"


def resolve_default_board_path() -> Path:
    for candidate in DEFAULT_BOARD_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_BOARD_CANDIDATES[0]


@dataclass(frozen=True)
class CuratedContract:
    role: str
    expiration: str
    contract: str
    option_type: str
    strike: float
    mid: float | None
    bid: float | None
    ask: float | None
    volume: int | None
    open_interest: int | None
    width_pct: float | None
    priority: str
    flags: str
    reason: str


@dataclass(frozen=True)
class CuratorPacket:
    schema: str
    generated_at_utc: str
    symbol: str
    headline: str
    stance: str
    target_strike: float | None
    underlying_price: float | None
    iv_heat: dict[str, Any]
    cockpit_context: dict[str, Any]
    focus_contracts: list[CuratedContract]
    noise_filters: list[str]
    watch_next: list[str]
    warnings: list[str]
    hey_guy_summary: dict[str, Any]
    artifact_refs: dict[str, str]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fmt_money(value: Any) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}"


def fmt_pct(value: Any) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.1f}%"


def fmt_ratio(value: Any) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}x"


def summarize_iv_heat(iv_heat: dict[str, Any]) -> dict[str, Any]:
    if not iv_heat:
        return {"available": False}
    reads = list(iv_heat.get("expiry_reads") or [])
    target_strike = safe_float(iv_heat.get("target_strike"))
    target_reads = [
        read for read in reads if safe_float(read.get("call_750_mid")) is not None
    ]
    first_event = next(
        (
            read
            for read in target_reads
            if read.get("harvest_window") == "event_crush_window"
        ),
        None,
    )
    month_time = [
        read
        for read in target_reads
        if read.get("harvest_window") == "event_plus_month_time"
    ]
    return {
        "available": True,
        "overall_heat_label": iv_heat.get("overall_heat_label"),
        "median_iv_rv13_ratio": iv_heat.get("median_iv_rv13_ratio"),
        "rv13_pct": (iv_heat.get("realized_vol") or {}).get("rv13_pct"),
        "nearest_event": iv_heat.get("nearest_event"),
        "days_to_nearest_event": iv_heat.get("days_to_nearest_event"),
        "target_strike": target_strike,
        "front_event_expiry": _expiry_summary(first_event),
        "month_time_expiries": [_expiry_summary(read) for read in month_time[:3]],
    }


def _expiry_summary(read: dict[str, Any] | None) -> dict[str, Any] | None:
    if not read:
        return None
    return {
        "expiration": read.get("expiration"),
        "dte_calendar": read.get("dte_calendar"),
        "atm_iv_pct": read.get("atm_iv_pct"),
        "iv_rv13_ratio": read.get("iv_rv13_ratio"),
        "heat_label": read.get("heat_label"),
        "target_call_mid": read.get("call_750_mid"),
        "target_call_iv_pct": read.get("call_750_iv_pct"),
        "target_call_open_interest": read.get("call_750_open_interest"),
        "harvest_window": read.get("harvest_window"),
    }


def summarize_cockpit(signal: dict[str, Any]) -> dict[str, Any]:
    stack = (signal.get("historical_refill_context") or {}).get(
        "active_refill_stack"
    ) or {}
    radar = signal.get("event_radar") or {}
    return {
        "available": bool(signal),
        "signal_ts": signal.get("ts"),
        "spot": signal.get("spot"),
        "setup_tag": signal.get("setup_tag"),
        "entry_setup_tag": signal.get("entry_setup_tag"),
        "entry_setup_bias": signal.get("entry_setup_bias"),
        "gamma_regime": signal.get("gamma_regime"),
        "event_headline": radar.get("headline"),
        "stack_label": stack.get("stack_label"),
        "active_stack_count": stack.get("active_count"),
        "nearest_refill_target": stack.get("nearest_target"),
        "highest_refill_target": stack.get("highest_target"),
    }


def curate_contracts(
    board: dict[str, Any],
    *,
    target_strike: float | None,
    underlying: float | None,
    downside_bias: bool = False,
) -> list[CuratedContract]:
    contracts = list(board.get("contracts") or [])
    calls = [row for row in contracts if row.get("option_type") == "call"]
    puts = [row for row in contracts if row.get("option_type") == "put"]
    focus: list[CuratedContract] = []
    if downside_bias:
        focus.extend(_best_liquid_puts(puts, underlying=underlying))
        focus.extend(_best_near_money_pressure(contracts, underlying=underlying))
        focus.extend(_best_target_calls(calls, target_strike=target_strike))
    else:
        focus.extend(_best_target_calls(calls, target_strike=target_strike))
        focus.extend(_best_near_money_pressure(contracts, underlying=underlying))
        focus.extend(_best_liquid_puts(puts, underlying=underlying))
    deduped: list[CuratedContract] = []
    seen: set[str] = set()
    for item in focus:
        key = item.contract
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:10]


def _best_target_calls(
    rows: list[dict[str, Any]], *, target_strike: float | None
) -> list[CuratedContract]:
    if target_strike is None:
        return []
    candidates = [
        row
        for row in rows
        if abs((safe_float(row.get("strike")) or 0) - target_strike) <= 2.5
    ]
    candidates.sort(
        key=lambda row: (_expiry_rank(row), -(safe_int(row.get("open_interest")) or 0))
    )
    return [
        _contract(row, "target-call", f"near the {target_strike:g} thesis strike")
        for row in candidates[:4]
    ]


def _best_near_money_pressure(
    rows: list[dict[str, Any]], *, underlying: float | None
) -> list[CuratedContract]:
    if underlying is None:
        return []
    candidates = [
        row
        for row in rows
        if abs(((safe_float(row.get("strike")) or 0) / underlying - 1.0) * 100.0) <= 0.8
    ]
    candidates.sort(
        key=lambda row: (
            -(safe_int(row.get("volume")) or 0),
            -(safe_int(row.get("open_interest")) or 0),
        )
    )
    return [
        _contract(
            row, "near-money-flow", "near spot; useful for live pressure/flow read"
        )
        for row in candidates[:3]
    ]


def _best_liquid_puts(
    rows: list[dict[str, Any]], *, underlying: float | None
) -> list[CuratedContract]:
    candidates = rows
    if underlying is not None:
        candidates = [
            row for row in rows if (safe_float(row.get("strike")) or 0) <= underlying
        ]
    candidates.sort(
        key=lambda row: (
            -(safe_int(row.get("volume")) or 0),
            -(safe_int(row.get("open_interest")) or 0),
        )
    )
    return [
        _contract(
            row, "downside-hedge", "liquid put/hedge line to watch if reclaim fails"
        )
        for row in candidates[:3]
    ]


def _expiry_rank(row: dict[str, Any]) -> tuple[str, float]:
    return (
        str(row.get("expiration") or "9999-99-99"),
        safe_float(row.get("strike")) or 0.0,
    )


def _contract(row: dict[str, Any], role: str, reason: str) -> CuratedContract:
    return CuratedContract(
        role=role,
        expiration=str(row.get("expiration") or ""),
        contract=str(row.get("contract_symbol") or ""),
        option_type=str(row.get("option_type") or ""),
        strike=safe_float(row.get("strike")) or 0.0,
        mid=safe_float(row.get("midpoint")),
        bid=safe_float(row.get("bid")),
        ask=safe_float(row.get("ask")),
        volume=safe_int(row.get("volume")),
        open_interest=safe_int(row.get("open_interest")),
        width_pct=safe_float(row.get("width_pct")),
        priority=str(row.get("manual_validation_priority") or ""),
        flags=str(row.get("rejection_flags") or "none"),
        reason=reason,
    )


def _is_downside_bias(cockpit: dict[str, Any]) -> bool:
    bias_text = " ".join(
        str(cockpit.get(key) or "")
        for key in ("entry_setup_bias", "setup_tag", "entry_setup_tag")
    ).lower()
    return any(token in bias_text for token in ("down", "bear", "put", "reversal down"))


def _resolve_reference_levels(
    cockpit: dict[str, Any],
    focus_contracts: list[CuratedContract],
    *,
    downside_bias: bool,
    target: float | None,
) -> tuple[float | None, float | None]:
    nearest = safe_float(cockpit.get("nearest_refill_target"))
    highest = safe_float(cockpit.get("highest_refill_target"))
    if nearest is None:
        roles = (
            {"downside-hedge"} if downside_bias else {"near-money-flow", "target-call"}
        )
        for item in focus_contracts:
            if item.role in roles:
                nearest = item.strike
                break
    if highest is None:
        highest = target or safe_float(cockpit.get("spot"))
    return nearest, highest


def build_watch_next(
    iv: dict[str, Any],
    cockpit: dict[str, Any],
    focus_contracts: list[CuratedContract],
) -> list[str]:
    target = iv.get("target_strike")
    nearest, highest = _resolve_reference_levels(
        cockpit,
        focus_contracts,
        downside_bias=_is_downside_bias(cockpit),
        target=safe_float(target),
    )
    spot = cockpit.get("spot")
    heat = iv.get("overall_heat_label")
    ratio = iv.get("median_iv_rv13_ratio")
    if _is_downside_bias(cockpit):
        steps = [
            f"Track rejection/failure under {fmt_money(nearest)} first; without that failure, the put tape can just be hedge noise.",
            f"Watch IV/RV13 {fmt_ratio(ratio)} ({heat}); cheap premium is still useless if downside flow loses urgency.",
            f"If near-money puts lose bid while spot stabilizes around {fmt_money(spot)}, downgrade the bearish read fast.",
        ]
    else:
        steps = [
            f"Track reclaim acceptance around {fmt_money(nearest)} then {fmt_money(target)}; without acceptance, target calls are only a thesis, not a trigger.",
            f"Watch IV/RV13 {fmt_ratio(ratio)} ({heat}); event-crush windows are premium-harvest risk.",
            f"Treat {fmt_money(highest)} as farther-off unfinished business unless {fmt_money(target)} accepts with strength.",
        ]
    if (iv.get("front_event_expiry") or {}).get(
        "harvest_window"
    ) == "event_crush_window":
        steps.append(
            "After FOMC/AAPL resolution, re-check the same focus line again; do not confuse cheaper premium with worse thesis."
        )
    return steps


def build_noise_filters() -> list[str]:
    return [
        "Ignore contracts flagged zero_or_tiny_market/missing_midpoint unless only studying lotto flow.",
        "Do not rank solely by volume; near-expiry volume can be decay/closing flow, not clean intent.",
        "Use broker-fresh quotes before evaluating any spread/debit; Yahoo/NERV is research-only.",
        "Front expiries can look cheap but may expire before the Apple/FOMC thesis has time to mature.",
    ]


def build_warnings(board: dict[str, Any], iv: dict[str, Any]) -> list[str]:
    warnings = [
        "Curator is descriptive only; approval_decision remains the only authority object.",
        "NERV free/public data requires broker-fresh confirmation before execution.",
    ]
    summary = board.get("summary") or {}
    if summary.get("data_mode"):
        warnings.append(f"NERV data mode: {summary.get('data_mode')}.")
    if iv.get("overall_heat_label") in {"hot", "very_hot"}:
        warnings.append(
            "IV/RV13 says premium is hot; entry timing must account for harvest risk."
        )
    return warnings


def build_headline(
    iv: dict[str, Any],
    cockpit: dict[str, Any],
    focus_contracts: list[CuratedContract],
) -> tuple[str, str]:
    target = iv.get("target_strike")
    heat = iv.get("overall_heat_label") or "unknown"
    ratio = fmt_ratio(iv.get("median_iv_rv13_ratio"))
    nearest_level, highest_level = _resolve_reference_levels(
        cockpit,
        focus_contracts,
        downside_bias=_is_downside_bias(cockpit),
        target=safe_float(target),
    )
    nearest = fmt_money(nearest_level)
    highest = fmt_money(highest_level)
    spot = fmt_money(cockpit.get("spot"))
    magnet_context = f", upside magnet {highest}" if highest_level is not None else ""
    if _is_downside_bias(cockpit):
        level_context = (
            f"failure watch {nearest} under spot {spot}"
            if nearest_level is not None
            else "failure line unavailable; require fresh price confirmation"
        )
        headline = (
            f"Curate NERV around downside hedge lines: IV/RV13 {ratio} ({heat}), "
            f"{level_context}{magnet_context}."
        )
    else:
        target_context = (
            f"{fmt_money(target)} calls"
            if target is not None
            else "call-side liquidity"
        )
        level_context = (
            f"reclaim path {nearest} → {fmt_money(target)}"
            if nearest_level is not None and target is not None
            else "reclaim path unavailable; do not infer a numeric trigger"
        )
        headline = (
            f"Curate NERV around {target_context}: IV/RV13 {ratio} ({heat}), "
            f"{level_context}{magnet_context}."
        )
    stance = (
        "wait_for_acceptance_or_harvest"
        if heat in {"hot", "very_hot"}
        else "watch_reclaim_path"
    )
    return headline, stance


def build_packet(
    *,
    board_path: Path | None = None,
    iv_heat_path: Path = DEFAULT_IV_HEAT,
    signal_path: Path = DEFAULT_SIGNAL,
) -> CuratorPacket:
    board_path = board_path or resolve_default_board_path()
    board = read_json(board_path)
    iv_heat_raw = read_json(iv_heat_path)
    signal = read_json(signal_path)
    iv = summarize_iv_heat(iv_heat_raw)
    cockpit = summarize_cockpit(signal)
    target = safe_float(iv.get("target_strike"))
    iv_underlying = safe_float(iv_heat_raw.get("underlying_price"))
    live_underlying = safe_float(cockpit.get("spot"))
    underlying = live_underlying if live_underlying is not None else iv_underlying
    downside_bias = _is_downside_bias(cockpit)
    focus_contracts = curate_contracts(
        board,
        target_strike=target,
        underlying=underlying,
        downside_bias=downside_bias,
    )
    headline, stance = build_headline(iv, cockpit, focus_contracts)
    warnings = build_warnings(board, iv)
    if (
        live_underlying is not None
        and iv_underlying is not None
        and abs(live_underlying - iv_underlying) >= 0.01
    ):
        warnings.append(
            "IV-heat underlying is stale or mismatched; displayed spot uses the live cockpit signal."
        )
    return CuratorPacket(
        schema="sharpedge.nerv_curator.v1",
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        symbol=str(iv_heat_raw.get("symbol") or "SPY"),
        headline=headline,
        stance=stance,
        target_strike=target,
        underlying_price=underlying,
        iv_heat=iv,
        cockpit_context=cockpit,
        focus_contracts=focus_contracts,
        noise_filters=build_noise_filters(),
        watch_next=build_watch_next(iv, cockpit, focus_contracts),
        warnings=warnings,
        hey_guy_summary=build_hey_guy_summary(
            headline=headline,
            stance=stance,
            setup_bias=(
                cockpit.get("entry_setup_bias")
                or cockpit.get("setup_tag")
                or "no clear options bias"
            ),
            target_strike=iv.get("target_strike"),
            heat_label=iv.get("overall_heat_label") or "unknown",
            iv_rv_ratio_text=fmt_ratio(iv.get("median_iv_rv13_ratio")),
            downside_bias=downside_bias,
            focus_contracts=[asdict(item) for item in focus_contracts],
            nearest_level=cockpit.get("nearest_refill_target"),
            spot=cockpit.get("spot"),
        ),
        artifact_refs={
            "nerv_board": str(board_path),
            "iv_heat": str(iv_heat_path),
            "signal": str(signal_path),
        },
    )


def write_packet(packet: CuratorPacket, json_path: Path, txt_path: Path) -> None:
    write_packet_files(packet, json_path, txt_path, render_text=render_text)


def render_text(packet: CuratorPacket) -> str:
    return render_packet_text(packet, fmt_money)


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate NERV into an operator read.")
    parser.add_argument(
        "--board",
        default="",
        help="Optional explicit NERV board path. Defaults to the freshest known board source.",
    )
    parser.add_argument("--iv-heat", default=str(DEFAULT_IV_HEAT))
    parser.add_argument("--signal", default=str(DEFAULT_SIGNAL))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT))
    args = parser.parse_args()

    packet = build_packet(
        board_path=Path(args.board) if args.board else None,
        iv_heat_path=Path(args.iv_heat),
        signal_path=Path(args.signal),
    )
    write_packet(packet, Path(args.json_out), Path(args.txt_out))
    print(json.dumps({"json": args.json_out, "txt": args.txt_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
