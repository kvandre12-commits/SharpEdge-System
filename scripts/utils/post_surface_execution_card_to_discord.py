#!/usr/bin/env python3
"""Post SharpEdge surface execution card to Discord.

Reads outputs/surface_execution_card.txt and posts a compact summary to the
configured Discord webhook.
"""

import json
import os
import urllib.request
from pathlib import Path

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
CARD_PATH = Path("outputs/surface_execution_card.txt")
JSON_PATH = Path("outputs/surface_execution_card.json")
MAX_LEN = 1800


def load_message():
    if JSON_PATH.exists():
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))

        lines = [
            "📡 SHARPEDGE SURFACE STATE",
            "",
            f"UNDERLYING: {payload.get('underlying')}",
            f"RISK POSTURE: {payload.get('risk_posture')}",
            "",
            f"EXHAUSTION: {payload.get('exhaustion_state')}",
            f"TRANSITION: {payload.get('transition_state')}",
            f"PARTICIPATION: {payload.get('participation_quality')}",
            "",
            "EXECUTION NOTE:",
            payload.get('execution_note', ''),
            "",
            "SYSTEM NOTE:",
            payload.get('system_note', ''),
        ]

        return "\n".join(lines)[:MAX_LEN]

    if CARD_PATH.exists():
        return CARD_PATH.read_text(encoding="utf-8")[:MAX_LEN]

    return None


def post(msg):
    body = json.dumps({"content": msg}).encode("utf-8")

    req = urllib.request.Request(
        WEBHOOK,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        print(f"discord status={resp.status}")


def main():
    if not WEBHOOK:
        print("DISCORD_WEBHOOK_URL missing; skipping")
        return

    msg = load_message()

    if not msg:
        print("No surface execution card found; skipping")
        return

    post(msg)
    print("Posted SharpEdge surface execution card to Discord")


if __name__ == "__main__":
    main()
