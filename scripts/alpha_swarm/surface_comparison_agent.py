#!/usr/bin/env python3
"""Compare frozen Paper Boy hypotheses with the live SharpEdge SPY surface."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.hypothesis_researcher import (  # noqa: E402
    MOMENTUM_THRESHOLD,
    VOLUME_THRESHOLD,
    VWAP_THRESHOLD,
)

SCHEMA = "sharpedge.paper_live_surface_comparison.v1"
EXPECTED_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN")
EVENT_ACTIONS = (
    "research_prefetch",
    "publish_eligibility",
    "publish_hypothesis",
    "option_prefetch",
    "publish_expression_review",
    "entry_prefetch",
    "exit_prefetch",
    "publish_receipt",
)
EASTERN = ZoneInfo("America/New_York")
DEFAULT_SIGNAL = ROOT / "outputs" / "signal.json"
DEFAULT_MANIFEST = ROOT / "outputs" / "alpha_swarm_phase1_manifest.json"
DEFAULT_PILOT_ROOT = ROOT / "outputs" / "alpha_swarm_pilot"
DEFAULT_REPORT_ROOT = DEFAULT_PILOT_ROOT / "surface_comparison"
DEFAULT_HTML = ROOT / "cockpit" / "paper_boy_compare.html"


def _parse_timestamp(value: Any, *, naive_tz=UTC) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=naive_tz) if parsed.tzinfo is None else parsed


def _read_json(path: Path, warnings: list[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"{label} is missing: {path}")
        return {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"{label} is unreadable: {type(exc).__name__}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"{label} must contain a JSON object")
        return {}
    return payload


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _direction(vs_vwap: float | None, momentum: float | None) -> str:
    if vs_vwap is None or momentum is None:
        return "unknown"
    if vs_vwap >= VWAP_THRESHOLD and momentum >= MOMENTUM_THRESHOLD:
        return "bullish"
    if vs_vwap <= -VWAP_THRESHOLD and momentum <= -MOMENTUM_THRESHOLD:
        return "bearish"
    return "mixed"


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_latest_publications(
    pilot_root: Path, warnings: list[str]
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Return one coherent latest session, selected by publication timestamps."""
    candidates: list[tuple[str, datetime, str, Path, dict[str, Any]]] = []
    for path in pilot_root.glob("????-??-??/*/phase3_hypothesis.json"):
        payload = _read_json(path, warnings, f"paper publication {path.name}")
        candidate = payload.get("candidate") or {}
        symbol = str(payload.get("symbol") or candidate.get("symbol") or "").upper()
        session_date = str(payload.get("session_date") or "")
        published = _parse_timestamp(
            payload.get("published_at") or candidate.get("published_at"), naive_tz=UTC
        )
        if (
            not session_date
            or symbol not in EXPECTED_SYMBOLS
            or published is None
            or payload.get("paper_only") is not True
            or payload.get("authoritative") is not False
            or payload.get("execution_permitted") is not False
        ):
            warnings.append(f"ignored invalid paper publication: {path}")
            continue
        candidates.append((session_date, published, symbol, path, payload))
    if not candidates:
        return None, {}
    session_date = max(item[0] for item in candidates)
    selected: dict[str, dict[str, Any]] = {}
    for _, published, symbol, path, payload in sorted(candidates):
        if payload.get("session_date") != session_date:
            continue
        previous = selected.get(symbol)
        if previous and previous["_published"] >= published:
            continue
        selected[symbol] = {
            "payload": payload,
            "path": str(path),
            "sha256": _file_sha256(path),
            "_published": published,
        }
    return session_date, selected


def _manifest_slots(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(slot.get("slot_id")): slot
        for slot in manifest.get("slots", [])
        if isinstance(slot, dict) and slot.get("slot_id")
    }


def _event_statuses(pilot_root: Path, slot_id: str) -> dict[str, str]:
    statuses = {}
    for action in EVENT_ACTIONS:
        path = pilot_root / "events" / f"{slot_id}__{action}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            statuses[action] = "pending"
            continue
        statuses[action] = str(payload.get("status") or "unknown")
    return statuses


def _paper_reason(decision: str, features: dict[str, float | None]) -> list[str]:
    if decision != "stand_down":
        return [f"frozen rule published {decision}"]
    reasons = []
    volume = features["volume_ratio"]
    if volume is None or volume < VOLUME_THRESHOLD:
        reasons.append(f"volume {volume or 0:.3f}x was below {VOLUME_THRESHOLD:.1f}x")
    direction = _direction(features["vs_vwap_pct"], features["momentum_15m_pct"])
    if direction == "mixed":
        reasons.append("VWAP and 15-minute momentum were not directionally aligned")
    return reasons or ["no directional branch satisfied every frozen gate"]


def _paper_rows(
    session_date: str | None,
    publications: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
    pilot_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for symbol in EXPECTED_SYMBOLS:
        source = publications.get(symbol)
        if not source:
            rows.append({"symbol": symbol, "available": False, "decision": "unknown"})
            continue
        payload = source["payload"]
        candidate = payload.get("candidate") or {}
        slot_id = str(payload.get("slot_id") or candidate.get("slot_id") or "")
        slot = slots.get(slot_id, {})
        values = candidate.get("feature_values") or {}
        features = {
            "vs_vwap_pct": _number(values.get("vs_vwap_pct")),
            "momentum_15m_pct": _number(values.get("momentum_15m_pct")),
            "volume_ratio": _number(values.get("volume_ratio")),
        }
        decision = str(candidate.get("decision") or "unknown")
        rows.append(
            {
                "symbol": symbol,
                "available": True,
                "session_date": session_date,
                "slot_id": slot_id,
                "decision": decision,
                "direction_at_prediction": _direction(
                    features["vs_vwap_pct"], features["momentum_15m_pct"]
                ),
                "prediction_ts": candidate.get("prediction_ts"),
                "published_at": candidate.get("published_at")
                or payload.get("published_at"),
                "exit_ts": slot.get("exit_ts"),
                "label_available_ts": slot.get("label_available_ts"),
                "features": features,
                "decision_reasons": _paper_reason(decision, features),
                "risk_cap_dollars": candidate.get("risk_cap_dollars"),
                "event_statuses": _event_statuses(pilot_root, slot_id),
                "source_path": source["path"],
                "source_sha256": source["sha256"],
            }
        )
    return rows


def _live_surface(signal: dict[str, Any]) -> dict[str, Any]:
    permission = signal.get("trade_permission") or {}
    conviction = permission.get("setup_conviction") or {}
    vs_vwap = _number(signal.get("vs_vwap"))
    momentum = _number(signal.get("mom15"))
    volume = _number(signal.get("vol_mult"))
    gate = str(permission.get("trade_gate") or "UNKNOWN").upper()
    return {
        "symbol": str(signal.get("symbol") or "SPY"),
        "signal_ts": signal.get("ts"),
        "spot": _number(signal.get("spot")),
        "vwap": _number(signal.get("vwap")),
        "vs_vwap_pct": vs_vwap,
        "momentum_15m_pct": momentum,
        "volume_multiple": volume,
        "volume_confirmed": volume is not None and volume >= VOLUME_THRESHOLD,
        "direction": _direction(vs_vwap, momentum),
        "location_lean": (
            "bearish"
            if vs_vwap is not None and vs_vwap <= -VWAP_THRESHOLD
            else (
                "bullish"
                if vs_vwap is not None and vs_vwap >= VWAP_THRESHOLD
                else "balanced"
            )
        ),
        "gamma_regime": signal.get("gamma_regime"),
        "pin": _number(signal.get("pin")),
        "max_pain": _number(signal.get("max_pain")),
        "trade_gate": gate,
        "trade_permission_score": permission.get("trade_permission_score"),
        "execution_permission_score": permission.get("execution_permission_score"),
        "execution_posture": ("open" if gate in {"GO", "TRADE", "ALLOW"} else "closed"),
        "setup_gate": conviction.get("setup_gate"),
        "setup_tag": conviction.get("setup_tag") or signal.get("entry_setup_tag"),
        "setup_bias": conviction.get("bias") or signal.get("entry_setup_bias"),
        "price_feed_stale": (signal.get("price_authority") or {}).get(
            "price_feed_stale"
        ),
    }


def _worker_summary(worker: dict[str, Any]) -> dict[str, Any]:
    pid = int(worker.get("pid") or 0)
    counts: dict[str, int] = {}
    for receipt in (worker.get("events") or {}).values():
        status = str((receipt or {}).get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "state": "running" if pid and Path(f"/proc/{pid}").exists() else "stopped",
        "pid": pid or None,
        "heartbeat_at": worker.get("heartbeat_at"),
        "event_counts_all_sessions": counts,
        "paper_only": worker.get("paper_only") is True,
        "execution_permitted": worker.get("execution_permitted") is True,
    }


def build_report(
    *,
    signal: dict[str, Any],
    manifest: dict[str, Any],
    publications: dict[str, dict[str, Any]],
    session_date: str | None,
    worker: dict[str, Any],
    pilot_root: Path,
    generated_at: datetime,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    paper_rows = _paper_rows(
        session_date, publications, _manifest_slots(manifest), pilot_root
    )
    paper_spy = next(row for row in paper_rows if row["symbol"] == "SPY")
    live = _live_surface(signal)
    paper_ts = _parse_timestamp(paper_spy.get("prediction_ts"), naive_tz=EASTERN)
    live_ts = _parse_timestamp(live.get("signal_ts"), naive_tz=EASTERN)
    generated = generated_at.astimezone(UTC)
    gap_minutes = (
        round((live_ts - paper_ts).total_seconds() / 60, 1)
        if paper_ts and live_ts
        else None
    )
    live_age_minutes = (
        round((generated - live_ts.astimezone(UTC)).total_seconds() / 60, 1)
        if live_ts
        else None
    )
    temporal_state = "timestamp_error"
    if live_age_minutes is not None and live_age_minutes > 15:
        temporal_state = "stale_live_signal"
    elif gap_minutes is not None and gap_minutes > 1:
        temporal_state = "later_live_snapshot"
    elif gap_minutes is not None:
        temporal_state = "same_window"
    paper_no_action = paper_spy.get("decision") == "stand_down"
    live_no_action = live["execution_posture"] == "closed"
    action_alignment = (
        "both_no_action"
        if paper_no_action and live_no_action
        else "different_action_posture"
    )
    paper_direction = paper_spy.get("direction_at_prediction", "unknown")
    live_direction = live["direction"]
    direction_change = f"{paper_direction}_to_{live_direction}"
    if action_alignment == "both_no_action" and live_direction == "mixed":
        headline = (
            f"Both stand down; live location leans {live['location_lean']}, "
            "but momentum and volume do not confirm."
        )
    elif action_alignment == "both_no_action":
        headline = (
            f"Both stand down; tape changed {direction_change.replace('_', ' ')}."
        )
    else:
        headline = f"Action posture differs; tape changed {direction_change.replace('_', ' ')}."
    current_event_counts: dict[str, int] = {}
    for row in paper_rows:
        for status in row.get("event_statuses", {}).values():
            current_event_counts[status] = current_event_counts.get(status, 0) + 1
    return {
        "schema": SCHEMA,
        "generated_at": generated.isoformat(),
        "mode": "observational_read_only",
        "headline": headline,
        "paper_surface": {
            "session_date": session_date,
            "symbols": paper_rows,
            "expected_symbol_count": len(EXPECTED_SYMBOLS),
            "available_symbol_count": sum(row["available"] for row in paper_rows),
            "current_session_event_counts": current_event_counts,
        },
        "live_sharpedge_surface": live,
        "spy_comparison": {
            "action_alignment": action_alignment,
            "direction_change": direction_change,
            "paper_decision": paper_spy.get("decision"),
            "live_execution_posture": live["execution_posture"],
            "paper_reason": paper_spy.get("decision_reasons", []),
            "live_reason": [
                f"trade gate is {live['trade_gate']}",
                f"live direction is {live_direction}",
                f"live location leans {live['location_lean']}",
                (
                    "live volume passes 1.2x"
                    if live["volume_confirmed"]
                    else "live volume is below 1.2x"
                ),
            ],
            "interpretation": (
                "Compare process state only. The live snapshot occurred after the "
                "frozen prediction and cannot revise or grade it."
            ),
        },
        "temporal_alignment": {
            "state": temporal_state,
            "paper_prediction_ts": paper_spy.get("prediction_ts"),
            "paper_exit_ts": paper_spy.get("exit_ts"),
            "live_signal_ts": live.get("signal_ts"),
            "prediction_to_live_minutes": gap_minutes,
            "live_age_minutes": live_age_minutes,
            "same_decision_window": temporal_state == "same_window",
        },
        "paper_worker": _worker_summary(worker),
        "data_quality": {
            "warnings": warnings,
            "live_signal_available": bool(signal),
            "manifest_available": bool(manifest),
            "paper_session_available": session_date is not None,
        },
        "safety": {
            "paper_only": True,
            "authoritative": False,
            "execution_permitted": False,
            "can_mutate_paper_artifacts": False,
            "can_override_approval_decision": False,
            "aggregate_score_computed": False,
            "hindsight_use": "comparison_only",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    live = report["live_sharpedge_surface"]
    comparison = report["spy_comparison"]
    temporal = report["temporal_alignment"]
    worker = report["paper_worker"]
    lines = [
        "# Paper Boy ↔ SharpEdge Surface Report",
        "",
        f"**{report['headline']}**",
        "",
        "> Observational and paper-only. This report cannot authorize execution, "
        "rewrite a frozen candidate, or compute the hidden aggregate score.",
        "",
        "## Direct SPY comparison",
        "",
        f"- Paper decision: **{comparison['paper_decision']}**",
        f"- Live posture: **{comparison['live_execution_posture']}** "
        f"(`{live['trade_gate']}` {live['trade_permission_score']})",
        f"- Tape change: **{comparison['direction_change']}**",
        f"- Paper → live gap: **{temporal['prediction_to_live_minutes']} minutes** "
        f"(`{temporal['state']}`)",
        f"- Live SPY: **${live['spot']:.2f}**, VWAP **${live['vwap']:.2f}**, "
        f"vs VWAP **{live['vs_vwap_pct']:+.3f}%**, momentum "
        f"**{live['momentum_15m_pct']:+.3f}%**, volume "
        f"**{live['volume_multiple']:.2f}×**",
        f"- Gamma: **{live['gamma_regime']}**, pin **{live['pin']}**, "
        f"max pain **{live['max_pain']}**",
        "",
        "### Why the surfaces differ",
        "",
        *[f"- Paper: {reason}" for reason in comparison["paper_reason"]],
        *[f"- Live: {reason}" for reason in comparison["live_reason"]],
        f"- {comparison['interpretation']}",
        "",
        "## Paper universe",
        "",
        "| Symbol | Decision | At prediction | vs VWAP | 15m mom | Volume |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["paper_surface"]["symbols"]:
        if not row["available"]:
            lines.append(f"| {row['symbol']} | unavailable | — | — | — | — |")
            continue
        features = row["features"]
        lines.append(
            f"| {row['symbol']} | {row['decision']} | "
            f"{row['direction_at_prediction']} | "
            f"{features['vs_vwap_pct']:+.3f}% | "
            f"{features['momentum_15m_pct']:+.3f}% | "
            f"{features['volume_ratio']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Paper Boy health",
            "",
            f"- State: **{worker['state']}** (PID `{worker['pid']}`)",
            f"- Heartbeat: `{worker['heartbeat_at']}`",
            "- Current-session event counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    report["paper_surface"]["current_session_event_counts"].items()
                )
            ),
            "- Historical event counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(worker["event_counts_all_sessions"].items())
            ),
            "",
            "## Safety boundary",
            "",
            "- No candidate mutation",
            "- No evaluator or aggregate-score computation",
            "- No broker or order authority",
            "- Approval decisions remain the only authority",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: dict[str, Any], markdown: str) -> str:
    body = escape(markdown)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Boy ↔ SharpEdge</title>
<style>
body{{background:#0b1018;color:#e8edf5;font:16px/1.5 system-ui;margin:0}}
main{{max-width:980px;margin:auto;padding:24px}} pre{{white-space:pre-wrap;background:#111a27;
padding:20px;border:1px solid #26364d;border-radius:12px}} .safe{{color:#8ce99a}}
</style></head><body><main><div class="safe">READ-ONLY • PAPER-ONLY • NO EXECUTION</div>
<pre>{body}</pre></main></body></html>"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    signal = _read_json(args.signal, warnings, "live signal")
    manifest = _read_json(args.manifest, warnings, "locked manifest")
    worker = _read_json(args.pilot_root / "worker_state.json", warnings, "worker state")
    session_date, publications = discover_latest_publications(args.pilot_root, warnings)
    report = build_report(
        signal=signal,
        manifest=manifest,
        publications=publications,
        session_date=session_date,
        worker=worker,
        pilot_root=args.pilot_root,
        generated_at=datetime.now(UTC),
        warnings=warnings,
    )
    markdown = render_markdown(report)
    _atomic_write(args.output_json, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.output_markdown, markdown)
    _atomic_write(args.output_html, render_html(report, markdown))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_REPORT_ROOT / "latest.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_REPORT_ROOT / "latest.md"
    )
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--interval-seconds", type=float, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_seconds < 0:
        raise SystemExit("--interval-seconds must be non-negative")
    while True:
        try:
            report = build_from_paths(args)
            print(
                f"surface report: {report['headline']} -> {args.output_markdown}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"surface report failed: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            if not args.interval_seconds:
                return 1
        if not args.interval_seconds:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
