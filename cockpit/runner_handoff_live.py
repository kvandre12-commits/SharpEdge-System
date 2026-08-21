"""Dedicated live runner-handoff surface for SharpEdge."""

from __future__ import annotations

from live_read_view import render_live_read_html
from setup_event_lifecycle import ACTIVE_EVENT_STATUSES

HANDOFF_TAG = "EXHAUSTION -> RUNNER HANDOFF"
RUNNER_DAY_TAG = "RUNNER DAY (wheee)"


def _is_active_status(status: object) -> bool:
    return str(status or "").lower() in ACTIVE_EVENT_STATUSES


def _dedupe_by_tag(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for item in items:
        tag = str((item or {}).get("tag") or "")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        unique.append(item)
    return unique


def _active_event_types(
    setup_events: list[dict[str, object]] | None,
    permission: dict[str, object] | None = None,
) -> set[str]:
    active = {
        str(event.get("event_type") or "")
        for event in (setup_events or [])
        if _is_active_status((event or {}).get("status"))
    }
    setup_conviction = (permission or {}).get("setup_conviction") or {}
    lifecycle = setup_conviction.get("event_lifecycle") or {}
    setup_tag = str(setup_conviction.get("setup_tag") or "")
    if setup_tag and _is_active_status(lifecycle.get("status")):
        active.add(setup_tag)
    return {tag for tag in active if tag}


def _synthetic_active_setup(
    permission: dict[str, object] | None = None,
) -> dict[str, object]:
    setup_conviction = (permission or {}).get("setup_conviction") or {}
    lifecycle = setup_conviction.get("event_lifecycle") or {}
    setup_tag = str(setup_conviction.get("setup_tag") or "")
    if not setup_tag or not _is_active_status(lifecycle.get("status")):
        return {}
    gate = str(setup_conviction.get("setup_gate") or "").upper()
    kind = "ok" if gate in {"ACTIONABLE", "EMERGING"} else "info"
    return {
        "tag": setup_tag,
        "bias": setup_conviction.get("bias") or "NEUTRAL",
        "kind": kind,
        "detail": setup_conviction.get("reason") or setup_tag,
    }


def _focus_setups(
    setups: list[dict[str, object]] | None,
    permission: dict[str, object] | None = None,
    setup_events: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    items = list(setups or [])
    active_types = _active_event_types(setup_events, permission)
    if not active_types:
        return items

    matched = [item for item in items if str(item.get("tag") or "") in active_types]
    synthetic = _synthetic_active_setup(permission)
    synthetic_tag = str(synthetic.get("tag") or "")
    if synthetic_tag and synthetic_tag in active_types:
        matched.append(synthetic)

    if HANDOFF_TAG in active_types:
        runner_context = [
            item for item in items if str(item.get("tag") or "") == RUNNER_DAY_TAG
        ]
        return _dedupe_by_tag([*runner_context, *matched]) or items

    focused = _dedupe_by_tag(matched)
    return focused or items


def render_runner_handoff_live_html(
    pa: dict[str, object],
    op: dict[str, object],
    lines: list[tuple[str, str, str]],
    setups: list[dict[str, object]] | None = None,
    permission: dict[str, object] | None = None,
    micro: dict[str, object] | None = None,
    magnitude: dict[str, object] | None = None,
    gp: dict[str, object] | None = None,
    permission_trend: dict[str, object] | None = None,
    edge_token_position: dict[str, object] | None = None,
    regime_refinement: dict[str, object] | None = None,
    weekly_context: dict[str, object] | None = None,
    monthly_context: dict[str, object] | None = None,
    stamp: str = "",
    setup_events: list[dict[str, object]] | None = None,
    timeframe_agreement: dict[str, object] | None = None,
    transition_pressure: dict[str, object] | None = None,
) -> str:
    focus_setups = _focus_setups(setups, permission, setup_events)
    return render_live_read_html(
        pa,
        op,
        lines,
        focus_setups,
        permission or {},
        micro or {},
        magnitude or {},
        gp or {},
        permission_trend or {},
        {},
        edge_token_position or {},
        regime_refinement or {},
        weekly_context,
        monthly_context,
        stamp,
        timeframe_agreement=timeframe_agreement,
        transition_pressure=transition_pressure,
    )


__all__ = ["render_runner_handoff_live_html"]
