from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import context_attachment as ctx  # noqa: E402


def _session_rows() -> list[tuple[int, float, float, float, float, int]]:
    return [
        (
            idx,
            100.0 + idx * 0.01,
            100.1 + idx * 0.01,
            99.9 + idx * 0.01,
            100.02 + idx * 0.01,
            1000,
        )
        for idx in range(20)
    ]


def test_context_attachment_always_returns_weekly_and_monthly(monkeypatch):
    monkeypatch.setattr(
        ctx, "fetch_weekly_context_rows", lambda: ([], {"source": "test"})
    )
    monkeypatch.setattr(
        ctx, "fetch_monthly_context_rows", lambda: ([], {"source": "test"})
    )

    packet = ctx.build_context_attachment(_session_rows(), spot=100.0)

    assert "weekly_context" in packet
    assert "monthly_context" in packet
    assert packet["weekly_context"]["context_available"] is True
    assert packet["monthly_context"]["context_available"] is True
    assert packet["weekly_context"]["legend"] == []
    assert packet["monthly_context"]["legend"] == []


def test_context_attachment_degrades_loudly_on_fetch_failure(monkeypatch):
    def fail_weekly():
        raise RuntimeError("weekly feed down")

    def fail_monthly():
        raise RuntimeError("monthly feed down")

    monkeypatch.setattr(ctx, "fetch_weekly_context_rows", fail_weekly)
    monkeypatch.setattr(ctx, "fetch_monthly_context_rows", fail_monthly)

    packet = ctx.build_context_attachment(_session_rows(), spot=100.0)

    assert packet["weekly_context"]["context_available"] is False
    assert packet["monthly_context"]["context_available"] is False
    assert "weekly feed down" in packet["weekly_context"]["detail"]
    assert "monthly feed down" in packet["monthly_context"]["detail"]
    assert packet["weekly_rows"] == []
    assert packet["monthly_rows"] == []
