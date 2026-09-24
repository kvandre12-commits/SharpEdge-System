from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from robinhood_social.market_context import attach_pending_market_context


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_social(db: Path, observations: list[tuple[str, str, str]]) -> None:
    with sqlite3.connect(db) as connection:
        connection.executescript(
            """
            CREATE TABLE post_observations (
                capture_id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                PRIMARY KEY (capture_id, post_id)
            );
            CREATE TABLE evaluation_eligibility (
                capture_id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                evaluation_scope TEXT NOT NULL,
                policy_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                effective_anchor_at_utc TEXT,
                PRIMARY KEY (
                    capture_id, post_id, evaluation_scope, policy_version
                )
            );
            CREATE TABLE claim_fields (
                capture_id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                status TEXT NOT NULL,
                value_json TEXT,
                PRIMARY KEY (capture_id, post_id, field_name)
            );
            """
        )
        for capture_id, post_id, anchor in observations:
            connection.execute(
                "INSERT INTO post_observations VALUES (?, ?)",
                (capture_id, post_id),
            )
            connection.execute(
                """
                INSERT INTO evaluation_eligibility VALUES (
                    ?, ?, 'market_context_join', 1, 'eligible', ?
                )
                """,
                (capture_id, post_id, anchor),
            )
            connection.execute(
                """
                INSERT INTO claim_fields VALUES (
                    ?, ?, 'symbol', 'observed', ?
                )
                """,
                (capture_id, post_id, json.dumps("SPY")),
            )


def _seed_market(db: Path) -> None:
    with sqlite3.connect(db) as connection:
        connection.executescript(
            """
            CREATE TABLE spy_bars_15m (
                ts TEXT NOT NULL,
                session_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                trade_count INTEGER,
                vwap REAL,
                PRIMARY KEY (symbol, ts)
            );
            CREATE TABLE regime_daily (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime_label TEXT,
                regime_ts TEXT
            );
            CREATE TABLE open_resolution_regime (
                snapshot_ts TEXT NOT NULL,
                session_date TEXT NOT NULL,
                underlying TEXT NOT NULL,
                open_regime_label TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO spy_bars_15m VALUES (
                '2026-09-10T14:00:00Z', '2026-09-10', 'SPY',
                499.0, 501.0, 498.0, 500.0, 1000, 50, 499.5
            )
            """
        )
        connection.execute(
            """
            INSERT INTO regime_daily VALUES (
                '2026-09-09', 'SPY', 'balanced', '2026-09-09T21:00:00Z'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO open_resolution_regime VALUES (
                '2026-09-10T14:05:00Z', '2026-09-10', 'SPY', 'accepted_open'
            )
            """
        )


def _field(connection, capture_id: str, field_name: str):
    return connection.execute(
        """
        SELECT status, value_json, source_record_time, selection_method,
               offset_seconds, quality_status, reason
        FROM market_context_fields
        WHERE capture_id = ? AND field_name = ?
        """,
        (capture_id, field_name),
    ).fetchone()


def test_only_completed_same_session_bar_is_visible(tmp_path):
    social_db = tmp_path / "social.db"
    market_db = tmp_path / "market.db"
    _seed_social(
        social_db,
        [
            ("inside-bar", "post-1", "2026-09-10T14:10:00Z"),
            ("after-bar", "post-2", "2026-09-10T14:16:00Z"),
        ],
    )
    _seed_market(market_db)
    market_hash_before = _hash(market_db)

    first = attach_pending_market_context(social_db, market_db)
    second = attach_pending_market_context(social_db, market_db)

    with sqlite3.connect(social_db) as connection:
        inside = _field(connection, "inside-bar", "underlying_price")
        completed = _field(connection, "after-bar", "underlying_price")
        vwap = _field(connection, "after-bar", "completed_bar_vwap")
        regime = _field(connection, "after-bar", "prior_session_regime")
        open_state = _field(connection, "after-bar", "open_resolution_state")
        opening_range = _field(
            connection, "after-bar", "opening_range_relationship"
        )
        snapshot = connection.execute(
            """
            SELECT research_family, source_database, overall_quality
            FROM market_context_snapshots WHERE capture_id = 'after-bar'
            """
        ).fetchone()

    assert first["attached_observations"] == 2
    assert second["attached_observations"] == 0
    assert first["market_truth_open_mode"] == "read_only_query_only"
    assert inside[0] == "unavailable"
    assert inside[1] is None
    assert inside[6] == "no_completed_same_session_bar_at_anchor"
    assert completed[0] == "observed"
    assert json.loads(completed[1]) == 500.0
    assert completed[2] == "2026-09-10T14:00:00Z"
    assert completed[3] == "latest_completed_same_session_15m_bar_v1"
    assert completed[4] == 60.0
    assert json.loads(vwap[1]) == 499.5
    assert json.loads(regime[1]) == "balanced"
    assert json.loads(open_state[1]) == "accepted_open"
    assert opening_range[0] == "unavailable"
    assert opening_range[5] == "unsupported_without_lookahead"
    assert snapshot == (
        "social_information_value",
        "spy_truth.db",
        "context_available",
    )
    assert _hash(market_db) == market_hash_before


def test_anchor_after_coverage_does_not_receive_old_price_or_regime(tmp_path):
    social_db = tmp_path / "social.db"
    market_db = tmp_path / "market.db"
    _seed_social(
        social_db,
        [("late-anchor", "post-1", "2026-10-10T14:16:00Z")],
    )
    _seed_market(market_db)

    attach_pending_market_context(social_db, market_db)

    with sqlite3.connect(social_db) as connection:
        price = _field(connection, "late-anchor", "underlying_price")
        regime = _field(connection, "late-anchor", "prior_session_regime")
        coverage = _field(connection, "late-anchor", "intraday_data_coverage")
        preexisting_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert price[0] == "stale"
    assert price[1] is None
    assert price[5] == "anchor_after_coverage"
    assert regime[0] == "stale"
    assert regime[1] is None
    assert coverage[5] == "anchor_after_coverage"
    assert not {"market_outcomes", "creator_scores", "rankings"} & preexisting_tables
