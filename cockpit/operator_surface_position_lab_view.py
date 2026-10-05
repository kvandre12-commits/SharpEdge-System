"""Render helpers for the operator-surface position lab block."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "n/a"))


def render_position_lab_card(
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
            "position lab",
            f'<div style="color:{mute}">No position lab artifact yet.</div>',
            cyan,
        )

    geometry = payload.get("geometry") or {}
    calendar = payload.get("calendar_context") or {}
    highlights = calendar.get("highlighted_expirations") or {}
    freshness = payload.get("freshness") or {}
    branches = payload.get("branches") or []
    preferred = next(
        (branch for branch in branches if branch.get("status") == "preferred_right_now"),
        branches[0] if branches else {},
    )
    signal_age = freshness.get("signal_minutes_old")
    signal_age_label = f"signal {signal_age}m old" if signal_age is not None else "signal age n/a"
    greek_plan = (preferred.get("greek_dollar_plan") or {}) if preferred else {}
    body = (
        f'<div style="font-size:18px;font-weight:bold;color:{fg}">'
        f'{_esc(payload.get("primary_posture"))}</div>'
        f'<div style="color:{mute};font-size:13px;margin-top:4px">'
        f'{_esc(payload.get("posture_reason"))}</div>'
        f'<div style="margin-top:8px">'
        f'{chip(calendar.get("selected_expiration") or "n/a", blue)}'
        f'{chip(calendar.get("selection_reason") or "n/a", amber)}'
        f'{chip(signal_age_label, cyan)}'
        "</div>"
        f'<div style="margin-top:10px;color:{mute};font-size:12px">'
        f'setup {_esc(geometry.get("setup_tag"))} • gamma {_esc(geometry.get("gamma_regime"))} '
        f'• dealer {_esc(geometry.get("dealer_state"))} • premium {_esc(geometry.get("premium_read"))}</div>'
        f'<div style="margin-top:10px;color:{fg};font-size:14px;font-weight:bold">'
        f'Preferred live branch: {_esc(preferred.get("structure_label") or "wait")}</div>'
        f'<div style="color:{mute};font-size:12px;margin-top:4px">'
        f'status {chip(preferred.get("status") or "n/a", status_color(preferred.get("status")))}'
        "</div>"
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Trigger</div>'
        f'<div style="color:{fg};font-size:13px">{_esc(preferred.get("trigger"))}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Invalidation</div>'
        f'<div style="color:{fg};font-size:13px">{_esc(preferred.get("invalidation"))}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Greek-$ quick map</div>'
        f'<div style="color:{fg};font-size:13px">'
        f'{html.escape(str(greek_plan.get("greeks_source"))) + ": " if greek_plan.get("greeks_source") else ""}'
        f'{_esc(greek_plan.get("delta_interpretation") or "wait branch has no live Greek map yet")}</div>'
        f'<div style="color:{mute};font-size:12px;margin-top:4px">'
        f'1R {_esc(greek_plan.get("approx_stock_move_for_1r"))} • '
        f'theta {_esc(greek_plan.get("theta_dollars_per_day"))}/day • '
        f'IV -5 {_esc(greek_plan.get("iv_down_5pt_pnl"))}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:11px">Calendar ladder</div>'
        + "".join(
            chip(item, blue)
            for item in [
                highlights.get("next_expiration"),
                highlights.get("weekly_anchor"),
                highlights.get("next_weekly_anchor"),
                highlights.get("monthlyish_anchor"),
            ]
            if item
        )
        + f'<div style="margin-top:8px;color:{mute};font-size:11px">Branch menu</div>'
        + list_block(
            [
                f'{branch.get("branch_id")}: {branch.get("structure_label")} — {branch.get("trigger")}'
                for branch in branches[:3]
            ],
            empty="no branches surfaced",
        )
        + f'<div style="margin-top:8px;color:{mute};font-size:11px">Available expirations</div>'
        + "".join(chip(item, blue) for item in (calendar.get("available_expirations") or [])[:8])
    )
    return card("position lab", body, status_color(preferred.get("status") or "watch_only"))


__all__ = ["render_position_lab_card"]
