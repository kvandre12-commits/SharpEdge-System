"""Background self-audit + weight tuner for the confluence-zone engine.

Watches confluence-zone snapshots, grades whether each zone was RESPECTED or
BROKEN from the future spot path, attributes that back to factor *kinds*, and
writes a bounded per-kind weight-multiplier overlay that ``confluence_zones.py``
consumes only when ``SHARPEDGE_CONFLUENCE_REALTIME_ADJUST=1``.

Never trades, never edits the Python weight tables. Advisory. Stateless: it
recomputes from the whole snapshot ledger each run (mirrors the spine auditor),
so there is no drift state to persist; ``gain`` + ``multiplier_bounds`` are the
learning-rate/damping. Spot-only, touch-and-resolve grading (v1); a bars-based
upgrade is a future, opt-in enhancement.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_COMPACT_TS = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$")

from realtime_audit_io import (
    cache_snapshot_paths,
    parse_timestamp,
    read_json,
    read_ledger,
    utc_now_iso,
    write_ledger,
)

ADJUSTMENT_SCHEMA = "sharpedge.confluence_zone_adjustments.v1"
SHADOW_AUTHORITY = "diagnostic_shadow_overlay"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_PATH = ROOT / "outputs" / "signal.json"
DEFAULT_CACHE_DIR = ROOT / "outputs" / "cockpit_artifact_cache"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "confluence_zone_audit"
DEFAULT_LEDGER_PATH = DEFAULT_OUTPUT_DIR / "snapshots.jsonl"
DEFAULT_ADJUSTMENT_PATH = ROOT / "outputs" / "confluence_zone_adjustments.json"

DEFAULT_HORIZON_SECONDS = 900
DEFAULT_MIN_SAMPLES = 12
DEFAULT_GAIN = 1.5
DEFAULT_MULTIPLIER_BOUNDS = (0.7, 1.3)
DEFAULT_BUFFER_BPS_FLOOR = 3.0   # min buffer as bps of spot
DEFAULT_BUFFER_FRAC = 0.25       # or a fraction of zone width, whichever larger


def snapshot_from_signal(signal: dict[str, Any], observed_at: str) -> dict[str, Any] | None:
    """Reduce a signal to a confluence snapshot, or None if it carries no zones."""
    cz = (signal or {}).get("confluence_zones")
    spot = signal.get("spot") if signal else None
    # Skip only PRE-FEATURE snapshots (no field) or unpriced ones. A snapshot with
    # the field but empty zones is kept: its spot still feeds the forward path used
    # to grade earlier zones.
    if cz is None or spot is None:
        return None
    zones = cz.get("zones") or []
    reduced = []
    for zone in zones:
        kinds = sorted({
            str(f.get("kind")) for f in zone.get("contributing_factors", []) if f.get("kind")
        })
        reduced.append({
            "zone_id": zone.get("zone_id"),
            "side": zone.get("side"),
            "zone_lo": zone.get("zone_lo"),
            "zone_hi": zone.get("zone_hi"),
            "conviction_band": zone.get("conviction_band"),
            "kinds": kinds,
        })
    return {
        "ts": observed_at,
        "spot": float(spot),
        "gamma_regime": cz.get("gamma_regime") or signal.get("gamma_regime"),
        "zones": reduced,
    }


def _zone_buffer(zone: dict[str, Any], spot: float, bps_floor: float, frac: float) -> float:
    width = float(zone.get("zone_hi", 0) or 0) - float(zone.get("zone_lo", 0) or 0)
    return max(bps_floor * spot / 10_000.0, frac * max(width, 0.0))


def _resolve_zone(
    zone: dict[str, Any], future_spots: list[float], bps_floor: float, frac: float
) -> str:
    lo = _num(zone.get("zone_lo"))
    hi = _num(zone.get("zone_hi"))
    side = zone.get("side")
    if lo is None or hi is None or not future_spots:
        return "UNTESTED"
    buffer = _zone_buffer(zone, future_spots[0], bps_floor, frac)
    touched = False
    for spot in future_spots:
        if not touched:
            if lo <= spot <= hi:
                touched = True
            continue
        cleared_up = spot >= hi + buffer
        cleared_down = spot <= lo - buffer
        if side == "support":
            if cleared_up:
                return "RESPECTED"
            if cleared_down:
                return "BROKEN"
        else:  # resistance
            if cleared_down:
                return "RESPECTED"
            if cleared_up:
                return "BROKEN"
    return "UNTESTED" if not touched else "UNRESOLVED"


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def grade_zone_outcomes(
    snapshots: list[dict[str, Any]],
    *,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    buffer_bps_floor: float = DEFAULT_BUFFER_BPS_FLOOR,
    buffer_frac: float = DEFAULT_BUFFER_FRAC,
) -> list[dict[str, Any]]:
    """Grade every origin zone against its forward spot path within the horizon."""
    dated = []
    for snap in snapshots:
        ts = parse_timestamp(snap.get("ts"))
        if ts is not None and snap.get("spot") is not None:
            dated.append((ts, snap))
    dated.sort(key=lambda item: item[0])

    outcomes: list[dict[str, Any]] = []
    for idx, (origin_ts, origin) in enumerate(dated):
        horizon_end = origin_ts.timestamp() + horizon_seconds
        future_spots = [
            float(snap["spot"])
            for ts, snap in dated[idx + 1:]
            if origin_ts.timestamp() < ts.timestamp() <= horizon_end
        ]
        for zone in origin.get("zones", []):
            result = _resolve_zone(zone, future_spots, buffer_bps_floor, buffer_frac)
            outcomes.append({
                "origin_ts": origin.get("ts"),
                "zone_id": zone.get("zone_id"),
                "side": zone.get("side"),
                "regime": origin.get("gamma_regime"),
                "conviction_band": zone.get("conviction_band"),
                "kinds": zone.get("kinds", []),
                "result": result,
            })
    return outcomes


def summarize_factor_edges(
    outcomes: list[dict[str, Any]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    gain: float = DEFAULT_GAIN,
    bounds: tuple[float, float] = DEFAULT_MULTIPLIER_BOUNDS,
) -> dict[str, Any]:
    """Per-kind respected-rate vs baseline -> bounded multiplier (sample-gated)."""
    tested = [o for o in outcomes if o["result"] in {"RESPECTED", "BROKEN"}]
    total = len(tested)
    baseline = (sum(o["result"] == "RESPECTED" for o in tested) / total) if total else 0.0

    per_kind: dict[str, list[bool]] = {}
    for outcome in tested:
        respected = outcome["result"] == "RESPECTED"
        for kind in set(outcome.get("kinds", [])):  # de-dup: one vote per zone per kind
            per_kind.setdefault(kind, []).append(respected)

    lo, hi = bounds
    adjustments: dict[str, Any] = {}
    for kind, votes in sorted(per_kind.items()):
        n = len(votes)
        rate = sum(votes) / n if n else 0.0
        lift = rate - baseline
        if n < min_samples:
            multiplier, action = 1.0, "observe (insufficient samples)"
        else:
            multiplier = round(max(lo, min(hi, 1.0 + gain * lift)), 4)
            action = "boost" if multiplier > 1 else ("cut" if multiplier < 1 else "hold")
        adjustments[kind] = {
            "multiplier": multiplier,
            "tested": n,
            "respected_rate": round(rate, 4),
            "baseline": round(baseline, 4),
            "lift": round(lift, 4),
            "action": action,
        }
    return {
        "baseline_respected_rate": round(baseline, 4),
        "total_tested": total,
        "adjustments": adjustments,
    }


def build_weight_overlay(
    summary: dict[str, Any],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    bounds: tuple[float, float] = DEFAULT_MULTIPLIER_BOUNDS,
) -> dict[str, Any]:
    adjustments = summary.get("adjustments") or {}
    active = any(a.get("multiplier", 1.0) != 1.0 for a in adjustments.values())
    return {
        "schema": ADJUSTMENT_SCHEMA,
        "generated_at": utc_now_iso(),
        "authority": SHADOW_AUTHORITY,
        "enabled": bool(active),
        "granularity": "factor_kind",
        "min_samples": min_samples,
        "baseline_respected_rate": summary.get("baseline_respected_rate", 0.0),
        "total_tested": summary.get("total_tested", 0),
        "multiplier_bounds": list(bounds),
        "adjustments": adjustments,
        "note": (
            "Consumed only when SHARPEDGE_CONFLUENCE_REALTIME_ADJUST=1. "
            "Advisory; never trades, never edits code weights."
        ),
    }


def render_text_report(overlay: dict[str, Any]) -> str:
    lines = [
        f"Confluence zone weight audit {overlay.get('generated_at', '')}",
        (
            f"  baseline respected rate {overlay.get('baseline_respected_rate')} "
            f"over {overlay.get('total_tested')} tested zones | enabled={overlay.get('enabled')}"
        ),
    ]
    for kind, adj in sorted((overlay.get("adjustments") or {}).items()):
        lines.append(
            f"  {kind:12s} x{adj.get('multiplier')}  "
            f"rate={adj.get('respected_rate')} (n={adj.get('tested')}) "
            f"lift={adj.get('lift')}  {adj.get('action')}"
        )
    if len(lines) == 2:
        lines.append("  no factor kinds tested yet")
    return "\n".join(lines)


def _observed_at_for(path: Path) -> str:
    """Capture time from the snapshot's ``<TS>`` cache dir, else file mtime."""
    match = _COMPACT_TS.match(path.parent.parent.name)
    if match:
        y, mo, d, h, mi, s = match.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}+00:00"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def run_once(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    adjustment_path: Path = DEFAULT_ADJUSTMENT_PATH,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    gain: float = DEFAULT_GAIN,
    bounds: tuple[float, float] = DEFAULT_MULTIPLIER_BOUNDS,
) -> dict[str, Any]:
    """Ingest new snapshots into the ledger, grade, and write the overlay + report."""
    ledger = {row["ts"]: row for row in read_ledger(ledger_path) if row.get("ts")}
    sources = list(cache_snapshot_paths(cache_dir))
    if signal_path.exists():
        sources.append(signal_path)
    for path in sources:
        signal = read_json(path)
        if not isinstance(signal, dict):
            continue
        snap = snapshot_from_signal(signal, _observed_at_for(path))
        if snap is not None:
            ledger[snap["ts"]] = snap

    snapshots = sorted(ledger.values(), key=lambda s: s.get("ts") or "")
    write_ledger(ledger_path, snapshots)

    outcomes = grade_zone_outcomes(snapshots, horizon_seconds=horizon_seconds)
    summary = summarize_factor_edges(outcomes, min_samples=min_samples, gain=gain, bounds=bounds)
    overlay = build_weight_overlay(summary, min_samples=min_samples, bounds=bounds)

    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    (output_dir / "latest.json").write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "latest.txt").write_text(render_text_report(overlay), encoding="utf-8")
    adjustment_path.parent.mkdir(parents=True, exist_ok=True)
    adjustment_path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")
    return overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    args = parser.parse_args(argv)

    def _cycle() -> None:
        overlay = run_once(
            horizon_seconds=args.horizon_seconds,
            min_samples=args.min_samples,
            gain=args.gain,
        )
        print(render_text_report(overlay))

    if not args.loop:
        _cycle()
        return 0
    while True:
        _cycle()
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
