#!/usr/bin/env python3
"""Render the simple options-walls box from outputs/signal.json.

Reads the canonical `sharpedge.signal.v1` artifact and writes a minimal,
phone-friendly walls page (spot vs put/call wall, pin, max pain) plus a short
text summary to stdout. Interpretation only.

Paths are env-overridable:
  SIGNAL_JSON  (default outputs/signal.json)
  WALLS_HTML   (default cockpit/walls.html — served alongside cockpit.html)
"""

from __future__ import annotations

import json
import os
import sys

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_COCKPIT_DIR = os.path.join(_REPO, "cockpit")
if _COCKPIT_DIR not in sys.path:
    sys.path.insert(0, _COCKPIT_DIR)
from walls_view import build_walls_html

SIGNAL_JSON = os.getenv("SIGNAL_JSON", os.path.join(_REPO, "outputs", "signal.json"))
WALLS_HTML = os.getenv("WALLS_HTML", os.path.join(_COCKPIT_DIR, "walls.html"))


def main(argv: list[str] | None = None) -> int:
    try:
        with open(SIGNAL_JSON, encoding="utf-8") as handle:
            signal = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"cannot read signal: {exc}", file=sys.stderr)
        return 1

    html_doc = build_walls_html(signal)
    os.makedirs(os.path.dirname(WALLS_HTML) or ".", exist_ok=True)
    with open(WALLS_HTML, "w", encoding="utf-8") as handle:
        handle.write(html_doc)

    spot = signal.get("spot")
    print(
        f"walls.html written -> {WALLS_HTML}\n"
        f"  SPY {spot} | put wall {signal.get('put_wall')} "
        f"<-> call wall {signal.get('call_wall')} | pin {signal.get('pin')} "
        f"| max_pain {signal.get('max_pain')} | {signal.get('gamma_regime')} gamma"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
