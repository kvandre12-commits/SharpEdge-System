from __future__ import annotations

from datetime import date

from scripts.alpha_swarm.contracts import parse_timestamp
from scripts.alpha_swarm.lock_manifest import build_manifest
from scripts.alpha_swarm.pilot_orchestrator import (
    PLAN_SCHEMA,
    build_plan,
    build_schedule,
)


def _manifest(universe=None):
    return build_manifest(
        run_id="pilot-plan-test",
        sessions=[date(2026, 8, 10)],
        universe=universe or ["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _now(value):
    return parse_timestamp(value, "now")


def test_schedule_expands_nine_ordered_events_per_slot():
    events = build_schedule(_manifest())
    assert len(events) == 9
    assert [event["stage_order"] for event in events] == list(range(10, 100, 10))
    assert events[0]["action"] == "acquire_research_snapshot"
    assert events[-1]["action"] == "publish_evaluation_receipt"
    assert events[3]["scheduled_at"] == "2026-08-10T14:46:00+00:00"


def test_pre_due_plan_names_exact_next_events():
    plan = build_plan(_manifest(), now=_now("2026-08-10T10:00:00-04:00"))
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["event_counts"] == {
        "completed": 0,
        "due": 0,
        "missed": 0,
        "pending": 9,
    }
    assert plan["next_scheduled_at"] == "2026-08-10T14:30:00+00:00"
    assert [event["action"] for event in plan["next_events"]] == [
        "acquire_research_snapshot",
        "publish_data_eligibility",
    ]


def test_exact_tick_returns_due_events_in_stage_then_slot_order():
    plan = build_plan(_manifest(["SPY", "QQQ"]), now=_now("2026-08-10T10:30:00-04:00"))
    due = plan["due_events"]
    assert [event["stage_order"] for event in due] == [10, 10, 20, 20]
    assert [event["symbol"] for event in due] == ["QQQ", "SPY", "QQQ", "SPY"]


def test_missed_events_never_become_catch_up_work():
    plan = build_plan(_manifest(), now=_now("2026-08-10T10:31:00-04:00"))
    assert plan["event_counts"]["missed"] == 2
    assert plan["event_counts"]["due"] == 0
    assert plan["due_events"] == []
    assert plan["catch_up_allowed"] is False
    assert plan["next_scheduled_at"] == "2026-08-10T14:45:00+00:00"


def test_completed_event_is_not_reissued_at_exact_tick():
    schedule = build_schedule(_manifest())
    completed = {schedule[0]["event_id"]}
    plan = build_plan(
        _manifest(),
        now=_now("2026-08-10T10:30:00-04:00"),
        completed_event_ids=completed,
    )
    assert plan["event_counts"]["completed"] == 1
    assert [event["action"] for event in plan["due_events"]] == [
        "publish_data_eligibility"
    ]


def test_every_event_is_paper_only_and_non_executable():
    plan = build_plan(_manifest(), now=_now("2026-08-10T10:00:00-04:00"))
    assert plan["background_scheduler_started"] is False
    assert plan["aggregate_score_computed"] is False
    assert plan["broker_action_allowed"] is False
    for event in plan["events"]:
        assert event["paper_only"] is True
        assert event["authoritative"] is False
        assert event["execution_permitted"] is False
        assert event["broker_action_allowed"] is False
        assert event["aggregate_score_computed"] is False


def test_conditions_prevent_unconditional_mark_acquisition():
    events = build_schedule(_manifest())
    conditions = {event["action"]: event["condition"] for event in events}
    assert conditions["acquire_entry_mark"] == "accepted_expression_only"
    assert conditions["acquire_exit_mark"] == "accepted_expression_only"
    assert conditions["acquire_option_snapshot"] == "directional_candidate_only"
