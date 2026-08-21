from __future__ import annotations

import json
from pathlib import Path
from typing import Any



def _fmt_money(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"${float(value):.2f}"



def _fmt_pct(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.3f}%"



def _render_gap_stats(prefix: str, packet: dict[str, Any]) -> list[str]:
    if not packet:
        return [f"- {prefix}: unavailable"]
    lines = [
        (
            f"- {prefix}: favorable {_fmt_pct(packet.get('probability_favorable_gap_pct'))} | "
            f"adverse {_fmt_pct(packet.get('probability_adverse_gap_pct'))} | "
            f"abs {_fmt_pct(packet.get('probability_abs_gap_pct'))}"
        ),
        (
            f"  weekday-aligned favorable {_fmt_pct(packet.get('weekday_probability_favorable_gap_pct'))} | "
            f"gap n={packet.get('sample_size') or 0}"
        ),
        f"  source: {packet.get('source')} ({packet.get('reason') or 'ok'})",
    ]
    if packet.get("filters"):
        lines.append(
            f"  proxy {packet.get('proxy_symbol')} filters {json.dumps(packet.get('filters'), sort_keys=True)} | "
            f"context matches {packet.get('context_match_count') or 0} | overlap {packet.get('overlap_sample_size') or 0}"
        )
    return lines



def render_text(payload: dict[str, Any]) -> str:
    contract = payload.get("contract") or {}
    open_ctx = payload.get("overnight_open") or {}
    empirical = open_ctx.get("empirical_gap_context") or {}
    conditioned = open_ctx.get("conditioned_gap_context") or {}
    lines = [
        "SHARPEDGE OVERNIGHT CARRY BRIEF",
        f"Created: {payload.get('generated_at_utc')}",
        (
            f"Contract: {contract.get('symbol')} {contract.get('strike')} "
            f"{str(contract.get('option_type') or '').upper()} exp {contract.get('expiration')}"
        ),
        (
            f"Spot {contract.get('spot')} | delta {contract.get('delta')} | gamma {contract.get('gamma')} | "
            f"theta/day {contract.get('theta_per_share_per_day')} | IV {contract.get('iv')}"
        ),
        "",
        "OPEN TOLL BOOTH",
        f"- overnight theta carry: {_fmt_money(open_ctx.get('theta_carry_contract'))}/contract",
        (
            f"- break-even favorable gap: {_fmt_money(open_ctx.get('break_even_move_dollars'))} "
            f"({_fmt_pct(open_ctx.get('break_even_move_pct'))})"
        ),
    ]
    lines.extend(_render_gap_stats("unconditional", empirical))
    lines.extend(_render_gap_stats("conditioned", conditioned))
    lines.extend(["", "GAP MAGNITUDE CONTEXT"])
    stats = empirical.get("magnitude_stats") or {}
    for label, pct_key, dollar_key in (
        ("median", "median_gap_pct", "median_gap_dollars"),
        ("p75", "p75_gap_pct", "p75_gap_dollars"),
        ("p90", "p90_gap_pct", "p90_gap_dollars"),
        ("p95", "p95_gap_pct", "p95_gap_dollars"),
        ("mean", "mean_gap_pct", "mean_gap_dollars"),
    ):
        lines.append(
            f"- {label}: {_fmt_pct(stats.get(pct_key))} | {_fmt_money(stats.get(dollar_key))}"
        )
    lines.extend(["", "OPEN P/L MAP"])
    for row in open_ctx.get("open_gap_pnl_scenarios") or []:
        lines.append(
            f"- {row.get('label')}: move {_fmt_money(row.get('move_dollars'))} ({_fmt_pct(row.get('move_pct'))}) | "
            f"up {_fmt_money(row.get('net_pnl_up_contract'))} | down {_fmt_money(row.get('net_pnl_down_contract'))}"
        )
    conditioned_rows = open_ctx.get("conditioned_open_gap_pnl_scenarios") or []
    if conditioned_rows:
        lines.extend(["", "CONDITIONED OPEN P/L MAP"])
        for row in conditioned_rows:
            lines.append(
                f"- {row.get('label')}: move {_fmt_money(row.get('move_dollars'))} ({_fmt_pct(row.get('move_pct'))}) | "
                f"up {_fmt_money(row.get('net_pnl_up_contract'))} | down {_fmt_money(row.get('net_pnl_down_contract'))}"
            )
    comparison_presets = open_ctx.get("comparison_presets") or []
    if comparison_presets:
        lines.extend(["", "PRESET COMPARISON BOARD"])
        for preset in comparison_presets:
            packet = preset.get("gap_context") or {}
            lines.extend(_render_gap_stats(str(preset.get("label") or "preset"), packet))
            first_move = next(
                (
                    row
                    for row in (preset.get("open_gap_pnl_scenarios") or [])
                    if row.get("label") == "median_abs_gap"
                ),
                None,
            )
            if first_move:
                lines.append(
                    f"  median abs gap map: move {_fmt_money(first_move.get('move_dollars'))} | "
                    f"up {_fmt_money(first_move.get('net_pnl_up_contract'))} | "
                    f"down {_fmt_money(first_move.get('net_pnl_down_contract'))}"
                )
    if open_ctx.get("open_iv_shock_scenarios"):
        lines.extend(["", "OPEN IV SHOCK MAP"])
        for row in open_ctx.get("open_iv_shock_scenarios") or []:
            lines.append(
                f"- IV {row.get('iv_points'):+} pts -> {_fmt_money(row.get('estimated_pnl_contract'))}"
            )
    lines.extend(["", "U-SHAPE INTRADAY BANDS"])
    for row in (payload.get("intraday") or {}).get("u_shape_bands") or []:
        lines.append(
            f"- {row.get('time_et')}: theta {_fmt_money(row.get('theta_carry_contract'))} | "
            f"BE {_fmt_money(row.get('break_even_move_dollars'))} ({_fmt_pct(row.get('break_even_move_pct'))}) | "
            f"1σ {_fmt_money(row.get('band_1sigma_dollars'))} | 95% {_fmt_money(row.get('band_95_dollars'))} | "
            f"+1σ {_fmt_money(row.get('pnl_plus_1sigma_contract'))} | -1σ {_fmt_money(row.get('pnl_minus_1sigma_contract'))}"
        )
    lines.extend(
        [
            "",
            "DOCTRINE",
            "- Open is the main theta toll booth.",
            "- Conditioned history matters more than decade-average wallpaper.",
            "- Spot can offset carry; IV and spread can still mug you in the parking lot.",
            f"- {payload.get('research_only_warning')}",
        ]
    )
    return "\n".join(lines) + "\n"



def write_outputs(payload: dict[str, Any], output_base: Path) -> tuple[Path, Path]:
    json_path = output_base.with_suffix(".json")
    txt_path = output_base.with_suffix(".txt")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(payload), encoding="utf-8")
    return json_path, txt_path


__all__ = ["render_text", "write_outputs"]
