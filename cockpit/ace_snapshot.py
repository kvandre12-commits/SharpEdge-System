from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PA_KEYS = (
    "spot",
    "vwap",
    "vs_vwap",
    "mom15",
    "vol_mult",
    "rng_pos",
    "balance_state",
    "position_in_balance",
    "balance_reference",
)

LEVEL_KEYS = ("ORH", "ORL", "PDH", "PDL", "PDC")
OP_KEYS = ("call_wall", "put_wall")
GP_KEYS = ("regime", "pin")


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source.get(key) for key in keys if key in source}


def build_ace_snapshot(
    rows: list[tuple[int, float, float, float, float, float]],
    pa: dict[str, Any],
    levels: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "sharpedge.ace_snapshot.v1",
        "bars": [list(bar) for bar in rows],
        "pa": _pick(pa, PA_KEYS),
        "levels": _pick(levels, LEVEL_KEYS),
        "op": _pick(op, OP_KEYS),
        "gp": _pick(gp, GP_KEYS),
    }


def write_ace_snapshot(
    rows: list[tuple[int, float, float, float, float, float]],
    pa: dict[str, Any],
    levels: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
    out_dir: str | Path,
) -> Path:
    payload = build_ace_snapshot(rows, pa, levels, op, gp)
    target_dir = Path(out_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "ace_snapshot.json"
    target_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target_path


__all__ = ["build_ace_snapshot", "write_ace_snapshot"]
