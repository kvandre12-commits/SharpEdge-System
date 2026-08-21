from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from runner_handoff_live import render_runner_handoff_live_html


def _permission(setup_tag: str, reason: str, gate: str = "ACTIONABLE") -> dict:
    return {
        "trade_gate": "CAUTION",
        "trade_permission_score": 74,
        "execution_permission_score": 74,
        "bias": "CALLS",
        "setup_conviction": {
            "setup_conviction_score": 79,
            "setup_gate": gate,
            "bias": "CALLS",
            "setup_tag": setup_tag,
            "reason": reason,
            "entry_gate": {"gate_id": "exhaustion_runner_handoff"},
            "context_gate": {"gate_id": "runner_day_directional_continuation"},
            "event_lifecycle": {
                "status": "confirmed",
                "first_seen_ts": "2026-07-09T10:42:00",
                "last_confirmed_ts": "2026-07-09T10:48:00",
                "observation_count": 3,
            },
        },
        "scores": {
            "structure_score": {
                "score": 81,
                "reason": "reversal structure promoted into continuation",
                "phase": "head",
                "phase_reason": "sequence is asserting and expanding",
            },
            "acceptance_score": {
                "score": 78,
                "reason": "multiple closes accepted above VWAP",
                "phase": "body",
                "phase_reason": "acceptance is established and holding",
            },
        },
    }


def _trend() -> dict:
    return {
        "points": [
            {"time": "10:30", "score": 58, "event_markers": ["DOWNSIDE EXHAUSTION CANDIDATE"]},
            {"time": "10:48", "score": 74, "event_markers": ["EXHAUSTION -> RUNNER HANDOFF CONFIRMED"]},
        ],
        "direction": "strengthening",
        "delta": 8,
        "largest_changes_since_last_update": [
            {"feature": "Structure", "delta": 12},
            {"feature": "Trend", "delta": 10},
        ],
    }


def _weekly_context() -> dict:
    return {
        "lookback_days": 5,
        "headline": "Promotion out of lower-edge exhaustion into mid-band continuation",
        "detail": "Spot reclaimed the inner carry shelf and is pressing higher.",
        "kind": "ok",
        "legend": [{"label": "H1 peak", "price": 746.1, "color": "#f85149"}],
    }


def _setup_events(*events: tuple[str, str]) -> list[dict]:
    return [
        {
            "event_type": event_type,
            "status": status,
        }
        for event_type, status in events
    ]


def _monthly_context() -> dict:
    return {
        "lookback_months": 6,
        "headline": "Monthly structure allows upside travel if continuation holds",
        "detail": "The handoff has room to stretch before colliding with the upper monthly rail.",
        "kind": "ok",
        "legend": [{"label": "upper rail", "price": 750.0, "color": "#f85149"}],
    }


def test_render_runner_handoff_live_html_goes_loud_when_handoff_active():
    html = render_runner_handoff_live_html(
        pa={"spot": 744.82, "day_chg": 0.23, "vwap": 742.95},
        op={"put_wall": 740.0, "call_wall": 750.0},
        lines=[("Bulls in control", "ok", "runner context")],
        setups=[
            {
                "tag": "RUNNER DAY (wheee)",
                "bias": "RIDE momentum - go directional, breakouts run",
                "kind": "warn",
                "detail": "negative gamma day type remains active in the background",
            },
            {
                "tag": "EXHAUSTION -> RUNNER HANDOFF",
                "bias": "CALLS (reversal promoted to runner)",
                "kind": "ok",
                "detail": "recent downside exhaustion reclaimed VWAP, cleared the pivot high, and now has momentum + volume confirmation",
            },
        ],
        permission=_permission(
            "EXHAUSTION -> RUNNER HANDOFF",
            "recent downside exhaustion reclaimed and expanded into continuation",
        ),
        micro={"ch_lo": 743.95, "ch_hi": 745.55},
        magnitude={"exp_move_realized_usd": 1.65},
        gp={"pin": 745.0, "regime": "negative"},
        permission_trend=_trend(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="10:48:00",
        setup_events=_setup_events(
            ("RUNNER DAY (wheee)", "confirmed"),
            ("EXHAUSTION -> RUNNER HANDOFF", "confirmed"),
            ("STICKY DAY (calm/chop)", "invalidated"),
        ),
    )

    assert "SharpEdge Live Read - SPY" in html
    assert "SharpEdge Runner Handoff Live - SPY" not in html
    assert "RUNNER HANDOFF STANDBY" not in html
    assert "LIVE RUNNER HANDOFF ACTIVE" not in html
    assert "PHASE PROMOTION • NOT JUST A VWAP FADE" in html
    assert "PHASE PROMOTION • CONTINUATION MANAGEMENT" in html
    assert "Manage this like continuation now — the fade has already handed off." in html
    assert "This setup has graduated beyond a simple VWAP fade." in html
    assert "RUNNER DAY (wheee)" in html
    assert "STICKY DAY (calm/chop)" not in html


def test_render_runner_handoff_live_html_synthesizes_active_handoff_when_raw_card_flickers():
    html = render_runner_handoff_live_html(
        pa={"spot": 744.82, "day_chg": 0.23, "vwap": 742.95},
        op={"put_wall": 740.0, "call_wall": 750.0},
        lines=[("Bulls in control", "ok", "runner context")],
        setups=[
            {
                "tag": "RUNNER DAY (wheee)",
                "bias": "RIDE momentum - go directional, breakouts run",
                "kind": "warn",
                "detail": "negative gamma day type remains active in the background",
            },
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
                "kind": "info",
                "detail": "stale context card",
            },
        ],
        permission=_permission(
            "EXHAUSTION -> RUNNER HANDOFF",
            "recent downside exhaustion reclaimed and expanded into continuation",
        ),
        micro={"ch_lo": 743.95, "ch_hi": 745.55},
        magnitude={"exp_move_realized_usd": 1.65},
        gp={"pin": 745.0, "regime": "negative"},
        permission_trend=_trend(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="10:49:00",
        setup_events=_setup_events(
            ("RUNNER DAY (wheee)", "confirmed"),
            ("EXHAUSTION -> RUNNER HANDOFF", "confirmed"),
            ("STICKY DAY (calm/chop)", "expired"),
        ),
    )

    assert "RUNNER DAY (wheee)" in html
    assert "EXHAUSTION -&gt; RUNNER HANDOFF" in html
    assert "STICKY DAY (calm/chop)" not in html
    assert "PHASE PROMOTION • NOT JUST A VWAP FADE" in html


def test_render_runner_handoff_live_html_uses_live_read_layout_when_no_handoff_active():
    html = render_runner_handoff_live_html(
        pa={"spot": 751.75, "day_chg": 0.71, "vwap": 748.96},
        op={"put_wall": 740.0, "call_wall": 752.0},
        lines=[("BULLS in control", "ok", "price above VWAP")],
        setups=[
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
                "kind": "info",
                "detail": "magnet below price; range fade only",
            }
        ],
        permission=_permission(
            "STICKY DAY (calm/chop)",
            "magnet below price; range fade only",
            gate="CONTEXT",
        ),
        micro={"ch_lo": 751.42, "ch_hi": 751.75},
        magnitude={"exp_move_realized_usd": 0.45},
        gp={"pin": 751.0, "regime": "positive"},
        permission_trend=_trend(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="14:06:45",
        setup_events=_setup_events(("STICKY DAY (calm/chop)", "confirmed")),
    )

    assert "SharpEdge Live Read - SPY" in html
    assert "SharpEdge Runner Handoff Live - SPY" not in html
    assert "RUNNER HANDOFF STANDBY" not in html
    assert "CURRENT LIVE SETUP CONTEXT" not in html
    assert "STICKY DAY (calm/chop)" in html
    assert "THE READ" in html
    assert "BULLS in control" in html
    assert "PHASE PROMOTION • NOT JUST A VWAP FADE" not in html
