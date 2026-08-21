from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from scripts.alpha_swarm.live_pilot import (
    _blocked_dependency,
    _sleep_until_next,
    event_schedule,
    run_worker,
    status,
)
from scripts.alpha_swarm.live_pilot_sources import (
    ACQUISITION_SCHEMA,
    fetch_options_capture,
    fetch_price_capture,
)
from scripts.alpha_swarm.lock_manifest import build_manifest


def _manifest():
    return build_manifest(
        run_id="live-pilot-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def test_schedule_prefetches_before_locked_publications():
    events = event_schedule(_manifest())
    assert len(events) == 8
    by_action = {event["action"]: event for event in events}
    assert by_action["research_prefetch"]["scheduled_at"].isoformat() == (
        "2026-08-10T14:28:00+00:00"
    )
    assert by_action["publish_eligibility"]["scheduled_at"].isoformat() == (
        "2026-08-10T14:30:00+00:00"
    )
    assert by_action["option_prefetch"]["scheduled_at"].isoformat() == (
        "2026-08-10T14:45:30+00:00"
    )


def test_yahoo_rows_become_hashed_provider_capture(monkeypatch):
    rows = [
        {
            "date": "2026-08-10",
            "minute_of_day": 570 + index,
            "open": 100.0,
            "high": 100.2,
            "low": 99.9,
            "close": 100.1,
            "volume": 100,
        }
        for index in range(20)
    ]
    rows.append(
        {
            **rows[-1],
            "open": rows[-1]["close"],
            "high": rows[-1]["close"],
            "low": rows[-1]["close"],
            "volume": 0,
        }
    )
    monkeypatch.setattr(
        "scripts.alpha_swarm.live_pilot_sources.market.fetch_yahoo_regular_session_chart_rows",
        lambda symbol: (rows, {"provider": "fixture"}),
    )
    capture = fetch_price_capture(
        "SPY",
        "2026-08-10",
        observed_at=datetime.fromisoformat("2026-08-10T10:30:00-04:00"),
    )
    assert capture["schema"] == ACQUISITION_SCHEMA
    assert capture["provider"] == "yahoo_chart_1m"
    assert len(capture["bars"]) == 20
    normalization = capture["source_metadata"]["normalization"]
    assert normalization["raw_session_row_count"] == 21
    assert normalization["terminal_placeholders_dropped"] == 1
    assert len(normalization["raw_session_rows_sha256"]) == 64
    assert capture["latest_data_ts"] == "2026-08-10T09:49:00-04:00"


def test_cboe_book_becomes_explicit_delayed_capture(monkeypatch):
    expiry = date(2026, 8, 21)
    raw = {
        "option": "SPY260821C00100000",
        "bid": 1.0,
        "ask": 1.1,
        "open_interest": 500,
        "volume": 50,
    }
    book = {expiry: {100.0: {"C": raw}}}
    monkeypatch.setattr(
        "scripts.alpha_swarm.live_pilot_sources.market.fetch_cboe_options_book",
        lambda symbol: (100.0, book, {"provider": "fixture"}),
    )
    observed = datetime(2026, 8, 10, 14, 29, tzinfo=UTC)
    capture = fetch_options_capture("SPY", "2026-08-10", observed_at=observed)
    assert capture["provider"] == "cboe_delayed_options_observed"
    assert capture["declared_feed_delay_minutes"] == 15
    assert capture["contracts"][0]["quote_ts"] == observed.isoformat()


def test_worker_before_first_event_writes_resumable_state_without_network(tmp_path):
    now = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    state = run_worker(
        _manifest(),
        output_root=tmp_path,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )
    assert state["events"] == {}
    assert state["paper_only"] is True
    assert state["execution_permitted"] is False
    report = status(tmp_path)
    assert report["alive"] is True
    assert report["paper_only"] is True


def test_late_worker_records_misses_without_backfill_or_network(tmp_path):
    now = datetime(2026, 8, 10, 14, 31, tzinfo=UTC)
    state = run_worker(
        _manifest(),
        output_root=tmp_path,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )
    statuses = {event["status"] for event in state["events"].values()}
    assert statuses == {"missed"}
    assert len(state["events"]) == 2
    assert all(
        "lateness tolerance" in event["error"] for event in state["events"].values()
    )
    assert not list(tmp_path.rglob("phase2_eligibility.json"))


def test_conflicting_duplicate_yahoo_bars_fail_closed(monkeypatch):
    row = {
        "date": "2026-08-10",
        "minute_of_day": 570,
        "open": 100.0,
        "high": 100.2,
        "low": 99.9,
        "close": 100.1,
        "volume": 100,
    }
    conflict = {**row, "close": 100.15, "volume": 50}
    monkeypatch.setattr(
        "scripts.alpha_swarm.live_pilot_sources.market.fetch_yahoo_regular_session_chart_rows",
        lambda symbol: ([row, conflict], {"provider": "fixture"}),
    )
    with pytest.raises(ValueError, match="conflicting Yahoo bars"):
        fetch_price_capture(
            "SPY",
            "2026-08-10",
            observed_at=datetime.fromisoformat("2026-08-10T10:30:00-04:00"),
        )


def test_dependency_cascade_is_blocked_not_executed():
    event = next(
        item
        for item in event_schedule(_manifest())
        if item["action"] == "option_prefetch"
    )
    hypothesis_id = f"{event['slot']['slot_id']}:publish_hypothesis"
    state = {"events": {hypothesis_id: {"status": "missed"}}}
    assert _blocked_dependency(event, state) == (hypothesis_id, "missed")


def test_adaptive_sleep_never_oversleeps_poll_interval():
    schedule = event_schedule(_manifest())
    now = datetime(2026, 8, 10, 14, 27, 59, 900000, tzinfo=UTC)
    waits = []
    _sleep_until_next(schedule, {"events": {}}, now, waits.append)
    assert waits == [0.1]


def test_restart_does_not_reissue_terminal_events(tmp_path):
    now = datetime(2026, 8, 10, 14, 31, tzinfo=UTC)
    first = run_worker(
        _manifest(),
        output_root=tmp_path,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )
    second = run_worker(
        _manifest(),
        output_root=tmp_path,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )
    assert second["events"] == first["events"]
    assert len(list((tmp_path / "events").glob("*.json"))) == len(first["events"])
