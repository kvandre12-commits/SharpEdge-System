#!/usr/bin/env python3
"""Render a compact institutional-style execution card.

This utility reads the latest exhaustion / transition state and turns it into
human-readable execution context for playbooks and Discord output.
"""

import json
import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def latest_state(con):
    return con.execute(
        """
        SELECT *
        FROM exhaustion_transition_state
        WHERE underlying=?
        ORDER BY session_date DESC, asof_ts DESC
        LIMIT 1
        """,
        (SYMBOL,),
    ).fetchone()


def risk_posture(state):
    ex = state["exhaustion_state"]
    part = state["participation_quality"]
    trans = state["transition_state"]

    if trans == "SURFACE_UNWIND_RISK":
        return "DEFENSIVE"
    if ex == "FAILED_CONTINUATION":
        return "REDUCE_SIZE"
    if part == "EMPTY_BREAKOUT_RISK":
        return "WAIT_FOR_CONFIRMATION"
    if ex == "HEALTHY_EXPANSION" and part == "HEALTHY_TREND":
        return "CONTROLLED_AGGRESSION"
    return "BALANCED"


def execution_note(state):
    posture = risk_posture(state)

    mapping = {
        "DEFENSIVE": "Dealer unwind / unstable transition detected. Avoid emotional participation and oversized entries.",
        "REDUCE_SIZE": "Continuation quality is deteriorating. Favor partials and responsive execution over chasing.",
        "WAIT_FOR_CONFIRMATION": "Breakout quality is weak. Require acceptance before directional commitment.",
        "CONTROLLED_AGGRESSION": "Trend participation is healthy. Pullback continuation setups remain valid while acceptance holds.",
        "BALANCED": "Auction is rotational. Avoid forcing directional conviction until structure resolves.",
    }

    return mapping[posture]


def render_card(state):
    posture = risk_posture(state)
    note = execution_note(state)

    lines = [
        "SHARPEDGE SURFACE EXECUTION CARD",
        "================================",
        f"UNDERLYING: {state['underlying']}",
        f"ASOF: {state['asof_ts']}",
        "",
        f"EXHAUSTION STATE: {state['exhaustion_state']}",
        f"TRANSITION STATE: {state['transition_state']}",
        f"PARTICIPATION: {state['participation_quality']}",
        f"EXHAUSTION SCORE: {float(state['exhaustion_score']):.2f}",
        "",
        f"RISK POSTURE: {posture}",
        "",
        "EXECUTION NOTE:",
        note,
        "",
        "SYSTEM NOTE:",
        state['note'],
    ]

    return "\n".join(lines)


def main():
    con = connect()
    state = latest_state(con)
    if not state:
        raise SystemExit("No exhaustion_transition_state rows found")

    card = render_card(state)

    txt_path = OUTDIR / "surface_execution_card.txt"
    txt_path.write_text(card, encoding="utf-8")

    json_payload = {
        "underlying": state["underlying"],
        "asof_ts": state["asof_ts"],
        "exhaustion_state": state["exhaustion_state"],
        "transition_state": state["transition_state"],
        "participation_quality": state["participation_quality"],
        "risk_posture": risk_posture(state),
        "execution_note": execution_note(state),
        "system_note": state["note"],
    }

    (OUTDIR / "surface_execution_card.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )

    print(card)


if __name__ == "__main__":
    main()
