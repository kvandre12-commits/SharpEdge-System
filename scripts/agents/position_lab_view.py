from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_text(payload: dict[str, Any]) -> str:
    geometry = payload.get("geometry") or {}
    freshness = payload.get("freshness") or {}
    calendar = payload.get("calendar_context") or {}
    highlights = calendar.get("highlighted_expirations") or {}
    lines = [
        "SHARPEDGE POSITION LAB",
        f"Created: {payload.get('generated_at_utc')}",
        f"Symbol: {payload.get('symbol')}",
        f"Primary posture: {payload.get('primary_posture')}",
        f"Reason: {payload.get('posture_reason')}",
        "",
        "CORE GEOMETRY",
        (
            f"- spot {geometry.get('spot'):.2f} | VWAP {geometry.get('vwap'):.2f} "
            f"({geometry.get('vs_vwap'):+.3f}%) | gamma {geometry.get('gamma_regime')} | dealer {geometry.get('dealer_state')}"
        ),
        (
            f"- pin {geometry.get('pin'):.2f} | call wall {geometry.get('call_wall'):.2f} "
            f"| put wall {geometry.get('put_wall'):.2f} | balance {geometry.get('balance_low'):.2f}-{geometry.get('balance_high'):.2f}"
        ),
        (
            f"- setup {geometry.get('setup_tag')} | trade gate {geometry.get('trade_gate')} "
            f"{geometry.get('trade_permission_score')} | volume {geometry.get('vol_mult'):.2f}x | premium {geometry.get('premium_read')}"
        ),
        (
            f"- quote freshness {freshness.get('quote_minutes_old_min')} to {freshness.get('quote_minutes_old_max')} minutes old "
            f"| expiration {freshness.get('expiration')}"
        ),
        f"- calendar reason: {calendar.get('selection_reason') or 'n/a'}",
        (
            "- calendar horizons: "
            f"next {highlights.get('next_expiration') or 'n/a'} | "
            f"weekly {highlights.get('weekly_anchor') or 'n/a'} | "
            f"next weekly {highlights.get('next_weekly_anchor') or 'n/a'} | "
            f"monthly-ish {highlights.get('monthlyish_anchor') or 'n/a'}"
        ),
        "",
        "BRANCHES",
    ]
    for branch in payload.get("branches") or []:
        lines.append(f"- {branch.get('branch_id')}: {branch.get('structure_label')}")
        lines.append(f"  status: {branch.get('status')}")
        lines.append(f"  trigger: {branch.get('trigger')}")
        lines.append(f"  invalidation: {branch.get('invalidation')}")
        lines.append(f"  thesis: {branch.get('thesis')}")
        lines.append(f"  caution: {branch.get('caution')}")
        pricing = branch.get("pricing") or {}
        if pricing:
            lines.append(
                "  pricing: "
                f"debit {pricing.get('debit')} | width {pricing.get('width')} | "
                f"max gain {pricing.get('max_gain')} | max loss {pricing.get('max_loss')} | breakeven {pricing.get('breakeven')}"
            )
        if branch.get("quote_quality"):
            lines.append(f"  quote quality: {branch.get('quote_quality')}")
        greek_plan = branch.get("greek_dollar_plan") or {}
        if greek_plan:
            lines.append(
                "  greek-$ plan: "
                f"[{greek_plan.get('greeks_source')}] {greek_plan.get('delta_interpretation')} | gamma {greek_plan.get('gamma_twitchiness')} "
                f"({greek_plan.get('net_gamma_share_change_per_1pt')} shares/$1) | "
                f"theta ${greek_plan.get('theta_dollars_per_day')}/day | vega ${greek_plan.get('vega_dollars_per_1iv')}/IV pt"
            )
            lines.append(
                "  dollar map: "
                f"1R move {greek_plan.get('approx_stock_move_for_1r')} | "
                f"25% theta burn {greek_plan.get('theta_days_to_25pct_decay')} days | "
                f"IV -5 pts ${greek_plan.get('iv_down_5pt_pnl')}"
            )
        lines.append("")
    boundary = payload.get("execution_boundary") or {}
    lines.extend(
        [
            "EXECUTION BOUNDARY",
            f"- trade allowed: {boundary.get('trade_allowed')} | broker order allowed: {boundary.get('broker_order_allowed')}",
            f"- contract decision: {boundary.get('decision')}",
            f"- blockers: {', '.join(boundary.get('blocking_reasons') or []) or 'none'}",
            f"- note: {boundary.get('note')}",
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
