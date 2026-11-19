from __future__ import annotations

from scripts.agents.operator_refill_context import summarize_refill_context


def test_signal_refill_context_is_compacted_without_permission() -> None:
    signal = {
        "historical_refill_context": {
            "available": True,
            "surface_generated_at": "2026-11-19T00:00:00+00:00",
            "active_refill_stack": {
                "stack_label": "double_dip_stack",
                "active_count": 2,
                "oldest_event_date": "2026-11-16",
                "latest_event_date": "2026-11-17",
                "nearest_target": 739.95,
                "highest_target": 754.37,
                "items": [
                    {
                        "mode": "intraday_dip",
                        "event_date": "2026-11-16",
                        "target": 754.37,
                        "elapsed_trading_days": 2,
                        "ignored_large_field": "nope",
                    }
                ],
                "historical_stack_surface": {
                    "overall": {"reference_count": 130}
                },
            },
        }
    }

    summary = summarize_refill_context(signal, {})

    assert summary["available"] is True
    assert summary["stack_label"] == "double_dip_stack"
    assert summary["nearest_target"] == 739.95
    assert summary["historical_stack_stats"]["reference_count"] == 130
    assert "ignored_large_field" not in summary["items"][0]
    assert summary["authority"] == "diagnostic_only"
    assert summary["permission_effect"] == "none"


def test_curator_is_a_bounded_fallback() -> None:
    curator = {
        "cockpit_context": {
            "stack_label": "triple_dip_stack",
            "active_stack_count": 3,
            "nearest_refill_target": 173.73,
            "highest_refill_target": 200.13,
        }
    }

    summary = summarize_refill_context({}, curator)

    assert summary["available"] is True
    assert summary["source"] == "nerv_curator.cockpit_context"
    assert summary["active_count"] == 3
    assert summary["permission_effect"] == "none"
