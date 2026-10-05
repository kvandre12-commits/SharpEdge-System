"""Render helpers for the operator-surface option-expression block."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "n/a"))



def render_option_expression_card(
    payload: dict[str, Any],
    *,
    card: Callable[[str, str, str], str],
    chip: Callable[[Any, str], str],
    list_block: Callable[[list[str], str], str],
    status_color: Callable[[Any], str],
    fg: str,
    mute: str,
    blue: str,
    amber: str,
    cyan: str,
) -> str:
    if not payload:
        return card(
            "option expression",
            f'<div style="color:{mute}">No option expression artifact yet.</div>',
            cyan,
        )

    doctrine = payload.get("expression_doctrine") or {}
    branches = [
        branch
        for branch in (payload.get("branch_expressions") or [])
        if branch.get("structure_family") != "no_forced_position"
    ]
    primary = branches[0] if branches else {}
    greek_plan = primary.get("greek_dollar_plan") or {}
    body = (
        f'<div style="font-size:16px;font-weight:bold;color:{fg}">'
        f'{_esc(doctrine.get("core_rule"))}</div>'
        f'<div style="margin-top:8px">'
        f'{chip(primary.get("structure_family") or "n/a", blue)}'
        f'{chip(greek_plan.get("greeks_source") or "n/a", blue)}'
        f'{chip(greek_plan.get("gamma_twitchiness") or "n/a", amber)}'
        f'{chip(greek_plan.get("theta_pressure") or "n/a", cyan)}'
        "</div>"
        f'<div style="margin-top:10px;color:{fg};font-size:14px;font-weight:bold">'
        f'Actionable branch: {_esc(primary.get("structure_label") or "wait")}</div>'
        f'<div style="color:{mute};font-size:12px;margin-top:4px">'
        f'{_esc(primary.get("expression_objective"))}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Greek-$ translation</div>'
        f'<div style="color:{fg};font-size:13px">{_esc(greek_plan.get("delta_interpretation"))}</div>'
        f'<div style="color:{mute};font-size:12px;margin-top:4px">'
        f'gamma { _esc(greek_plan.get("net_gamma_share_change_per_1pt")) } shares/$1 • '
        f'theta {_esc(greek_plan.get("theta_dollars_per_day"))}/day • '
        f'vega {_esc(greek_plan.get("vega_dollars_per_1iv"))}/IV pt</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Risk map</div>'
        f'<div style="color:{fg};font-size:13px">'
        f'1R move {_esc(greek_plan.get("approx_stock_move_for_1r"))} • '
        f'25% theta burn {_esc(greek_plan.get("theta_days_to_25pct_decay"))} days • '
        f'IV -5 pts {_esc(greek_plan.get("iv_down_5pt_pnl"))}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Thinking order</div>'
        + list_block(list(doctrine.get("thinking_order") or []), empty="no doctrine order")
    )
    return card("option expression", body, status_color(primary.get("status") or "watch_only"))


__all__ = ["render_option_expression_card"]
