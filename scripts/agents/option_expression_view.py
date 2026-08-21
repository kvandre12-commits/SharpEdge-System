from __future__ import annotations

import json
from pathlib import Path
from typing import Any



def _fmt_money(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"${float(value):.2f}"



def render_text(payload: dict[str, Any]) -> str:
    doctrine = payload.get("expression_doctrine") or {}
    market = payload.get("market_hypothesis") or {}
    lines = [
        "SHARPEDGE OPTION EXPRESSION",
        f"Created: {payload.get('generated_at_utc')}",
        f"Symbol: {payload.get('symbol')}",
        f"Core rule: {doctrine.get('core_rule')}",
        "",
        "MARKET HYPOTHESIS",
        (
            f"- setup {market.get('setup_tag')} | gamma {market.get('gamma_regime')} | "
            f"dealer {market.get('dealer_state')} | premium {market.get('premium_read')}"
        ),
        (
            f"- spot {market.get('spot')} | pin {market.get('pin')} | "
            f"call wall {market.get('call_wall')} | put wall {market.get('put_wall')}"
        ),
        "",
        "BRANCH EXPRESSIONS",
    ]
    for branch in payload.get("branch_expressions") or []:
        plan = branch.get("greek_dollar_plan") or {}
        lines.extend(
            [
                f"- {branch.get('branch_id')}: {branch.get('structure_label')}",
                f"  objective: {branch.get('expression_objective')}",
                f"  delta plan: [{plan.get('greeks_source')}] {plan.get('delta_interpretation')}",
                (
                    f"  gamma/theta/vega: {plan.get('gamma_twitchiness')} gamma "
                    f"({plan.get('net_gamma_share_change_per_1pt')} shares/$1), "
                    f"theta {_fmt_money(plan.get('theta_dollars_per_day'))}/day, "
                    f"vega {_fmt_money(plan.get('vega_dollars_per_1iv'))}/IV pt"
                ),
                (
                    f"  risk map: 1R move {plan.get('approx_stock_move_for_1r')} | "
                    f"25% theta burn {plan.get('theta_days_to_25pct_decay')} days | "
                    f"IV -5 pts {_fmt_money(plan.get('iv_down_5pt_pnl'))}"
                ),
                f"  trigger: {branch.get('trigger')}",
                f"  invalidation: {branch.get('invalidation')}",
                "",
            ]
        )
    boundary = payload.get("execution_boundary") or {}
    lines.extend(
        [
            "EXECUTION BOUNDARY",
            f"- decision: {boundary.get('decision')} | trade allowed: {boundary.get('trade_allowed')}",
            f"- blockers: {', '.join(boundary.get('blocking_reasons') or []) or 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"



def write_outputs(payload: dict[str, Any], output_base: Path) -> tuple[Path, Path]:
    json_path = output_base.with_suffix('.json')
    txt_path = output_base.with_suffix('.txt')
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    txt_path.write_text(render_text(payload), encoding='utf-8')
    return json_path, txt_path


__all__ = ['render_text', 'write_outputs']
