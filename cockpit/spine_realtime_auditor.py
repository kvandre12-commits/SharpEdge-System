"""Real-time diagnostic calibration loop for the cockpit execution spine.

This watches signal snapshots, scores later price outcomes, and writes a small
runtime adjustment overlay. It never places trades and never edits Python code.
If the cockpit wants to consume the overlay, it must opt in via
SHARPEDGE_SPINE_REALTIME_ADJUST=1.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_PATH = ROOT / "outputs" / "signal.json"
DEFAULT_CACHE_DIR = ROOT / "outputs" / "cockpit_artifact_cache"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "spine_realtime_audit"
DEFAULT_ADJUSTMENT_PATH = ROOT / "outputs" / "spine_realtime_adjustments.json"
DIRECTIONAL_BIASES = {"CALLS", "PUTS"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _feature_map(permission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = (permission.get("bucket_conditioned_spine") or {}).get("features")
    if not isinstance(features, list):
        features = (
            (permission.get("execution_hierarchy") or {}).get("core_spine") or {}
        ).get("features", [])
    by_name = {}
    for row in features or []:
        name = str((row or {}).get("name") or "")
        if name in CORE_EXECUTION_SPINE_PART_NAMES:
            by_name[name] = {
                "score": int((row or {}).get("score") or 0),
                "bias": str((row or {}).get("bias") or "NEUTRAL"),
                "weight": float((row or {}).get("weight") or 0.0),
                "reason": str((row or {}).get("reason") or ""),
            }
    return by_name


def snapshot_from_signal(
    signal: dict[str, Any], observed_at: datetime | None = None
) -> dict[str, Any] | None:
    permission = signal.get("trade_permission") or {}
    spot = signal.get("spot") or signal.get("last") or signal.get("price")
    if spot is None:
        return None
    ts = _parse_timestamp(signal.get("ts")) or observed_at
    if ts is None:
        return None
    features = _feature_map(permission)
    if not features:
        return None
    stamp = ts.astimezone(timezone.utc).isoformat()
    return {
        "schema": "sharpedge.spine_realtime_snapshot.v1",
        "snapshot_id": f"{stamp}|{float(spot):.4f}",
        "ts": stamp,
        "spot": float(spot),
        "gate": permission.get("trade_gate"),
        "score": int(permission.get("trade_permission_score") or 0),
        "bias": str(permission.get("bias") or "NEUTRAL"),
        "bucket": (permission.get("market_day") or {}).get("bucket"),
        "features": features,
        "score_spine_role": (permission.get("authority_self_audit") or {}).get(
            "score_spine_role", "diagnostic_advisory"
        ),
    }


def _cache_snapshot_paths(cache_dir: Path) -> list[Path]:
    if not cache_dir.exists():
        return []
    paths = []
    for child in sorted(cache_dir.iterdir()):
        signal_path = child / "outputs" / "signal.json"
        if child.is_dir() and child.name != "latest" and signal_path.exists():
            paths.append(signal_path)
    return paths


def _load_cached_snapshots(cache_dir: Path) -> list[dict[str, Any]]:
    snapshots = []
    for signal_path in _cache_snapshot_paths(cache_dir):
        signal = _read_json(signal_path)
        if not signal:
            continue
        observed_at = _parse_timestamp(signal_path.parent.parent.name)
        snapshot = snapshot_from_signal(signal, observed_at=observed_at)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_ledger(path: Path, snapshots: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = {str(row.get("snapshot_id")): row for row in snapshots}
    ordered = sorted(unique.values(), key=lambda row: str(row.get("ts") or ""))
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in ordered) + "\n",
        encoding="utf-8",
    )


def _direction_success(bias: str, move_bps: float, min_move_bps: float) -> bool | None:
    if abs(move_bps) < min_move_bps:
        return None
    if bias == "CALLS":
        return move_bps > 0
    if bias == "PUTS":
        return move_bps < 0
    return None


def evaluate_snapshots(
    snapshots: list[dict[str, Any]], horizon_seconds: int, min_move_bps: float
) -> list[dict[str, Any]]:
    ordered = sorted(snapshots, key=lambda row: str(row.get("ts") or ""))
    parsed = [(row, _parse_timestamp(row.get("ts"))) for row in ordered]
    evaluated = []
    for idx, (origin, origin_ts) in enumerate(parsed):
        if origin_ts is None:
            continue
        future = None
        for candidate, candidate_ts in parsed[idx + 1 :]:
            if candidate_ts is None:
                continue
            if (candidate_ts - origin_ts).total_seconds() >= horizon_seconds:
                future = candidate
                break
        if not future:
            continue
        start = float(origin.get("spot") or 0.0)
        end = float(future.get("spot") or 0.0)
        if start <= 0 or end <= 0:
            continue
        move_bps = ((end - start) / start) * 10_000
        trade_success = _direction_success(
            str(origin.get("bias") or "NEUTRAL"), move_bps, min_move_bps
        )
        feature_results = []
        for name, feature in (origin.get("features") or {}).items():
            bias = str(feature.get("bias") or "NEUTRAL")
            score = int(feature.get("score") or 0)
            if bias not in DIRECTIONAL_BIASES or score < 58:
                continue
            success = _direction_success(bias, move_bps, min_move_bps)
            if success is None:
                continue
            feature_results.append(
                {
                    "name": name,
                    "bias": bias,
                    "score": score,
                    "weight": float(feature.get("weight") or 0.0),
                    "success": bool(success),
                }
            )
        evaluated.append(
            {
                "origin_id": origin.get("snapshot_id"),
                "origin_ts": origin.get("ts"),
                "future_ts": future.get("ts"),
                "horizon_seconds": horizon_seconds,
                "start_spot": start,
                "future_spot": end,
                "move_bps": round(move_bps, 3),
                "trade_bias": origin.get("bias"),
                "trade_success": trade_success,
                "score": origin.get("score"),
                "gate": origin.get("gate"),
                "bucket": origin.get("bucket"),
                "features": feature_results,
            }
        )
    return evaluated


def summarize_feature_edges(
    evaluations: list[dict[str, Any]], min_samples: int
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        for feature in evaluation.get("features") or []:
            buckets[str(feature.get("name"))].append(
                {**feature, "move_bps": float(evaluation.get("move_bps") or 0.0)}
            )
    summary = {}
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        rows = buckets.get(name, [])
        wins = sum(1 for row in rows if row.get("success"))
        count = len(rows)
        hit_rate = wins / count if count else None
        avg_move = sum(float(row["move_bps"]) for row in rows) / count if count else 0.0
        if count < min_samples:
            action = "observe"
            delta = 0.0
        elif hit_rate is not None and hit_rate >= 0.58:
            action = "increase_weight_shadow"
            delta = 0.01
        elif hit_rate is not None and hit_rate <= 0.42:
            action = "decrease_weight_shadow"
            delta = -0.01
        else:
            action = "hold"
            delta = 0.0
        summary[name] = {
            "samples": count,
            "wins": wins,
            "losses": count - wins,
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            "avg_forward_move_bps": round(avg_move, 3),
            "action": action,
            "weight_delta": delta,
        }
    return summary


def build_adjustment_overlay(
    feature_summary: dict[str, dict[str, Any]], min_samples: int
) -> dict[str, Any]:
    adjustments = {}
    for name, row in feature_summary.items():
        delta = float(row.get("weight_delta") or 0.0)
        if delta == 0.0:
            continue
        adjustments[name] = {
            "weight_delta": delta,
            "reason": (
                f"{row.get('action')} from {row.get('samples')} samples; "
                f"hit_rate={row.get('hit_rate')}"
            ),
            "samples": row.get("samples"),
            "hit_rate": row.get("hit_rate"),
        }
    return {
        "schema": "sharpedge.spine_realtime_adjustments.v1",
        "generated_at": _utc_now_iso(),
        "authority": "diagnostic_shadow_overlay",
        "enabled": bool(adjustments),
        "min_samples": min_samples,
        "max_abs_weight_delta": 0.03,
        "adjustments": adjustments,
        "note": (
            "Consumed only when SHARPEDGE_SPINE_REALTIME_ADJUST=1. "
            "Does not grant final execution authority."
        ),
    }


def render_text_report(audit: dict[str, Any]) -> str:
    lines = [
        "SharpEdge spine realtime auditor",
        f"generated: {audit.get('generated_at')}",
        f"snapshots: {audit.get('snapshot_count')} | evaluated: {audit.get('evaluation_count')}",
        f"horizon: {audit.get('horizon_seconds')}s | min move: {audit.get('min_move_bps')} bps",
        "",
        "feature adjustments:",
    ]
    for name, row in (audit.get("feature_summary") or {}).items():
        lines.append(
            f"- {name}: {row.get('action')} delta={row.get('weight_delta'):+.2f} "
            f"samples={row.get('samples')} hit={row.get('hit_rate')} "
            f"avg_move={row.get('avg_forward_move_bps')}bps"
        )
    lines.append("")
    lines.append(
        "authority: diagnostic shadow overlay only; approval/operator still wins."
    )
    return "\n".join(lines) + "\n"


def run_once(
    *,
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    adjustment_path: Path = DEFAULT_ADJUSTMENT_PATH,
    horizon_seconds: int = 300,
    min_move_bps: float = 3.0,
    min_samples: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "snapshots.jsonl"
    snapshots = _read_ledger(ledger_path)
    snapshots.extend(_load_cached_snapshots(cache_dir))
    live_signal = _read_json(signal_path)
    if live_signal:
        live_snapshot = snapshot_from_signal(
            live_signal, observed_at=datetime.now(timezone.utc)
        )
        if live_snapshot:
            snapshots.append(live_snapshot)
    _write_ledger(ledger_path, snapshots)
    snapshots = _read_ledger(ledger_path)
    evaluations = evaluate_snapshots(snapshots, horizon_seconds, min_move_bps)
    feature_summary = summarize_feature_edges(evaluations, min_samples)
    overlay = build_adjustment_overlay(feature_summary, min_samples)
    audit = {
        "schema": "sharpedge.spine_realtime_audit.v1",
        "generated_at": _utc_now_iso(),
        "snapshot_count": len(snapshots),
        "evaluation_count": len(evaluations),
        "horizon_seconds": horizon_seconds,
        "min_move_bps": min_move_bps,
        "min_samples": min_samples,
        "feature_summary": feature_summary,
        "adjustment_overlay_path": str(adjustment_path),
        "authority": "diagnostic_only",
        "note": "Audits and writes shadow adjustments; final authority remains approval/operator.",
    }
    (output_dir / "latest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "latest.txt").write_text(render_text_report(audit), encoding="utf-8")
    adjustment_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adjustment-path", type=Path, default=DEFAULT_ADJUSTMENT_PATH)
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--min-move-bps", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    while True:
        audit = run_once(
            signal_path=args.signal,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            adjustment_path=args.adjustment_path,
            horizon_seconds=args.horizon_seconds,
            min_move_bps=args.min_move_bps,
            min_samples=args.min_samples,
        )
        print(render_text_report(audit))
        if not args.loop:
            return 0
        time.sleep(max(1, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
