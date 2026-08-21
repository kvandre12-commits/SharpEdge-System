"""Under-the-hood auction audit view for the SharpEdge cockpit."""

from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#7d8590"
SURFACE = "#161b22"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any, digits: int = 2, prefix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{prefix}{value:.{digits}f}"
    return "n/a" if value is None else str(value)


def _chip(label: str, value: Any, color: str = BLUE) -> str:
    return (
        f'<span style="display:inline-block;padding:3px 7px;margin:2px 5px 2px 0;'
        f'border:1px solid {color};border-radius:999px;color:{color};font-size:11px">'
        f"{_esc(label)}: {_esc(value)}</span>"
    )


def _row(title: str, value: str, detail: str = "", color: str = FG) -> str:
    detail_html = (
        f'<div style="color:#adbac7;font-size:11px;margin-top:3px;{WRAP}">{_esc(detail)}</div>'
        if detail
        else ""
    )
    return (
        f'<div style="padding:7px 0;border-bottom:1px solid #30363d">'
        f'<div style="color:{MUTE};font-size:10px;letter-spacing:.06em;font-weight:bold">{_esc(title)}</div>'
        f'<div style="color:{color};font-size:13px;font-weight:bold;margin-top:2px;{WRAP}">{value}</div>'
        f"{detail_html}</div>"
    )


def _feature_delta_rows(permission_trend: dict[str, Any]) -> str:
    changes = permission_trend.get("largest_changes_since_last_update") or []
    if not changes:
        return _row(
            "WHAT CHANGED", "No feature-score delta on the last refresh", "", MUTE
        )
    items = []
    for item in changes[:5]:
        delta = int(item.get("delta") or 0)
        color = GREEN if delta > 0 else RED
        arrow = "▲" if delta > 0 else "▼"
        items.append(
            f'<li style="margin:3px 0;color:{color}">{arrow} {_esc(item.get("feature") or "feature")} {delta:+d}</li>'
        )
    return _row(
        "WHAT CHANGED",
        f'<ul style="margin:4px 0 0 16px;padding:0">{"".join(items)}</ul>',
        "Permission trend explains whether the read is strengthening, weakening, or flat.",
        CYAN,
    )


def _setup_transition_rows(permission_trend: dict[str, Any]) -> str:
    transitions = permission_trend.get("setup_transitions_since_last_update") or []
    if not transitions:
        return _row("SETUP LIFECYCLE", "No new setup lifecycle transition", "", MUTE)
    items = []
    for item in transitions[:5]:
        items.append(
            f'<li style="margin:3px 0;color:{CYAN}">{_esc(item.get("label") or item.get("event_type") or "setup")}</li>'
        )
    return _row(
        "SETUP LIFECYCLE",
        f'<ul style="margin:4px 0 0 16px;padding:0">{"".join(items)}</ul>',
        "These are the setup-state changes the cockpit observed, not manual annotations.",
        CYAN,
    )


def _validation_row(
    decision_receipt: dict[str, Any], gap_fill_edge: dict[str, Any]
) -> str:
    outcome = decision_receipt.get("outcome") or {}
    reached = outcome.get("target_reached")
    if reached is not None:
        target_text = f"target reached: {bool(reached)}"
        detail = (
            f"max excursion {_fmt(outcome.get('max_excursion'))}; "
            f"session high {_fmt(outcome.get('session_high'), prefix='$')}; "
            f"session low {_fmt(outcome.get('session_low'), prefix='$')}"
        )
        return _row(
            "VALIDATION", _esc(target_text), detail, GREEN if reached else AMBER
        )

    if gap_fill_edge.get("available"):
        text = (
            f"gap fill prior n={gap_fill_edge.get('n', 'n/a')} / "
            f"fill rate {_fmt((gap_fill_edge.get('fill_rate') or 0) * 100, 1)}%"
        )
        detail = str(
            gap_fill_edge.get("path_mix_text") or gap_fill_edge.get("story") or ""
        )
        return _row("VALIDATION", _esc(text), detail, PURPLE)

    return _row("VALIDATION", "Outcome validation not attached yet", "", MUTE)


def render_under_hood_audit_block(
    *,
    permission: dict[str, Any] | None = None,
    permission_trend: dict[str, Any] | None = None,
    decision_receipt: dict[str, Any] | None = None,
    transition_pressure: dict[str, Any] | None = None,
    auction_context: dict[str, Any] | None = None,
    open_resolution: dict[str, Any] | None = None,
    dealer_positioning: dict[str, Any] | None = None,
    regime_read: dict[str, Any] | None = None,
    gap_fill_edge: dict[str, Any] | None = None,
) -> str:
    """Render compact receipts so Kurtis can audit the auction logic."""
    permission = permission or {}
    permission_trend = permission_trend or {}
    decision_receipt = decision_receipt or {}
    transition_pressure = transition_pressure or {}
    auction_context = auction_context or {}
    open_resolution = open_resolution or {}
    dealer_positioning = dealer_positioning or {}
    regime_read = regime_read or {}
    gap_fill_edge = gap_fill_edge or {}

    score = permission.get("execution_permission_score") or permission.get(
        "trade_permission_score", "n/a"
    )
    gate = permission.get("trade_gate") or decision_receipt.get("gate") or "n/a"
    bias = permission.get("bias") or decision_receipt.get("bias") or "NEUTRAL"
    delta = permission_trend.get("delta")
    delta_text = f"{delta:+d}" if isinstance(delta, int) else "n/a"
    direction = permission_trend.get("direction") or "new"
    pressure_score = transition_pressure.get("transition_pressure_score")
    pressure_state = transition_pressure.get("transition_state") or "n/a"

    auction_bits = "".join(
        [
            _chip("inherited auction", auction_context.get("bucket") or "n/a", PURPLE),
            _chip(
                "auction confidence", auction_context.get("confidence", "n/a"), PURPLE
            ),
            _chip(
                "open resolution",
                open_resolution.get("open_regime_label") or "n/a",
                CYAN,
            ),
            _chip("dealer", dealer_positioning.get("dealer_state") or "n/a", BLUE),
            _chip("gamma", dealer_positioning.get("gamma_regime") or "n/a", BLUE),
            _chip(
                "daily regime age", f"{regime_read.get('stale_days', 'n/a')}d", AMBER
            ),
        ]
    )
    rows = [
        _row(
            "FINAL AUTHORITY SNAPSHOT",
            _esc(f"{gate} / {score} / {bias}"),
            "This is the execution spine output. Diagnostic rows below explain it; they do not override it.",
            GREEN if str(gate).upper() == "ACTIONABLE" else AMBER,
        ),
        _row(
            "PERMISSION TREND",
            _esc(f"{direction} / delta {delta_text}"),
            "Trend is direction of travel; execution permission is the absolute gate.",
            CYAN,
        ),
        _feature_delta_rows(permission_trend),
        _setup_transition_rows(permission_trend),
        _row(
            "TRANSITION PRESSURE",
            _esc(
                f"{pressure_state} / {pressure_score if pressure_score is not None else 'n/a'}"
            ),
            str(transition_pressure.get("reason") or "No transition pressure packet."),
            PURPLE,
        ),
        _row(
            "AUCTION CONTEXT STACK",
            auction_bits,
            str(auction_context.get("story") or ""),
            BLUE,
        ),
        _row(
            "OPEN / DEALER RECEIPTS",
            _esc(str(open_resolution.get("story") or "No open-resolution story.")),
            str(dealer_positioning.get("story") or "No dealer-positioning story."),
            BLUE,
        ),
        _validation_row(decision_receipt, gap_fill_edge),
    ]
    return (
        f'<details open style="margin:12px 0;border:1px solid {PURPLE};border-radius:8px;background:#0d1117;padding:10px">'
        f'<summary style="cursor:pointer;color:{PURPLE};font-size:13px;font-weight:bold">'
        "UNDER THE HOOD: AUCTION LOGIC AUDIT</summary>"
        f'<div style="color:{MUTE};font-size:11px;margin:6px 0 8px;{WRAP}">'
        "Receipts for why the cockpit read shifted: score, feature deltas, setup lifecycle, transition pressure, inherited auction, open resolution, dealer gamma/OI, and validation context.</div>"
        f"{''.join(rows)}"
        "</details>"
    )


__all__ = ["render_under_hood_audit_block"]
