#!/usr/bin/env python3
"""Publish conservative paper marks and frozen-evaluator receipts after Phase 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    PAPER_MARK_SCHEMA,
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
    source_bundle_sha256,
    validate_candidate,
)
from scripts.alpha_swarm.evaluator import (  # noqa: E402
    evaluate_candidate,
    rejection_receipt,
    verify_evaluator_source_lock,
)
from scripts.alpha_swarm.lock_manifest import evaluator_source_paths  # noqa: E402
from scripts.alpha_swarm.skeptic_veto_agent import (  # noqa: E402
    PUBLICATION_SCHEMA as PHASE5_SCHEMA,
    build_review,
)

MARK_SNAPSHOT_SCHEMA = "sharpedge.alpha_swarm.paper_mark_snapshot.v1"
PUBLICATION_SCHEMA = "sharpedge.alpha_swarm.paper_mark_receipt_publication.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_MARK_FIELDS = frozenset(
    {
        "utility",
        "performance",
        "score",
        "alpha_score",
        "rank",
        "ret_1d",
        "return_prediction_to_exit",
        "broker",
        "route",
        "order_id",
    }
)


def _artifact_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_float(value: Any, field: str, *, allow_zero: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ValueError(
            f"{field} must be {'non-negative' if allow_zero else 'positive'}"
        )
    return number


def _walk_forbidden(value: Any, *, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_MARK_FIELDS:
                failures.append(f"forbidden field {path}.{key}")
            failures.extend(_walk_forbidden(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden(child, path=f"{path}[{index}]"))
    return failures


def _base(phase5: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "schema": PUBLICATION_SCHEMA,
        "run_id": phase5.get("run_id"),
        "manifest_sha256": phase5.get("manifest_sha256"),
        "evaluator_source_sha256": phase5.get("evaluator_source_sha256"),
        "slot_id": phase5.get("slot_id"),
        "session_date": phase5.get("session_date"),
        "symbol": phase5.get("symbol"),
        "published_at": now.isoformat(),
        "phase5_review_sha256": _artifact_sha256(phase5),
        "publisher_source_sha256": _source_sha256(Path(__file__)),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "aggregate_score_computed": False,
    }


def _validate_phase5_chain(
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase5: dict[str, Any],
    *,
    manifest: dict[str, Any] | None,
    option_snapshot: dict[str, Any] | None,
) -> None:
    if phase5.get("schema") != PHASE5_SCHEMA:
        raise ValueError(f"Phase 5 schema must be {PHASE5_SCHEMA}")
    if phase5.get("paper_only") is not True or phase5.get("authoritative") is not False:
        raise ValueError("Phase 5 review must remain paper-only and non-authoritative")
    if phase5.get("execution_permitted") is not False:
        raise ValueError("Phase 5 review must remain non-executable")
    expected_source = _source_sha256(Path(__file__).with_name("skeptic_veto_agent.py"))
    if phase5.get("skeptic_source_sha256") != expected_source:
        raise ValueError("Phase 5 skeptic source SHA256 does not match current source")
    if phase5.get("phase3_publication_sha256") != _artifact_sha256(phase3):
        raise ValueError("Phase 5 does not reference exact Phase 3 publication")
    if phase5.get("phase4_publication_sha256") != _artifact_sha256(phase4):
        raise ValueError("Phase 5 does not reference exact Phase 4 publication")
    review_time = parse_timestamp(phase5.get("reviewed_at"), "reviewed_at")
    replay = build_review(
        phase3,
        phase4,
        now=review_time,
        manifest=manifest,
        option_snapshot=option_snapshot,
    )
    if canonical_json(replay) != canonical_json(phase5):
        raise ValueError("Phase 5 review does not match deterministic replay")


def _candidate(phase3: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    candidate = phase3.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Phase 3 candidate is required for this Phase 5 state")
    validate_candidate(manifest, candidate)
    return candidate


def _mark_failure_receipt(
    manifest: dict[str, Any], phase5: dict[str, Any], reason: str
) -> dict[str, Any]:
    return rejection_receipt(manifest, str(phase5["slot_id"]), reason)


def _source_ref(source: dict[str, Any], name: str) -> str:
    if not str(source.get("provider") or "").strip():
        raise ValueError(f"{name}.provider is required")
    if not SHA256_RE.fullmatch(str(source.get("source_sha256") or "")):
        raise ValueError(f"{name}.source_sha256 is invalid")
    ref = str(source.get("source_ref") or "").strip()
    if not ref:
        raise ValueError(f"{name}.source_ref is required")
    return ref


def _build_mark(
    manifest: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    snapshot: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    candidate = _candidate(phase3, manifest)
    expression = phase4.get("expression")
    if not isinstance(expression, dict):
        raise ValueError(
            "accepted Phase 5 review requires the exact Phase 4 expression"
        )
    failures = _walk_forbidden(snapshot, path="mark_snapshot")
    if failures:
        raise ValueError(f"paper mark snapshot contains forbidden fields: {failures}")
    if snapshot.get("schema") != MARK_SNAPSHOT_SCHEMA:
        raise ValueError(f"paper mark snapshot schema must be {MARK_SNAPSHOT_SCHEMA}")
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("authoritative") is not False
    ):
        raise ValueError(
            "paper mark snapshot must remain paper-only and non-authoritative"
        )
    if snapshot.get("execution_permitted") is not False:
        raise ValueError("paper mark snapshot must remain non-executable")
    expected = {
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": candidate["slot_id"],
        "symbol": candidate["symbol"],
        "session_date": phase4["session_date"],
        "candidate_sha256": payload_sha256(candidate),
        "expression_sha256": _artifact_sha256(expression),
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            raise ValueError(f"paper mark snapshot {field} does not match locked chain")
    slots = {slot["slot_id"]: slot for slot in manifest["slots"]}
    slot = slots[candidate["slot_id"]]
    label_time = parse_timestamp(slot["label_available_ts"], "label_available_ts")
    if now != label_time:
        raise ValueError("receipt publication must occur at exact label_available_ts")
    if parse_timestamp(snapshot.get("captured_at"), "captured_at") != now:
        raise ValueError(
            "paper mark snapshot captured_at must equal label_available_ts"
        )

    entry = snapshot.get("entry") or {}
    exit_mark = snapshot.get("exit") or {}
    if parse_timestamp(
        entry.get("observed_at"), "entry.observed_at"
    ) != parse_timestamp(slot["entry_ts"], "entry_ts"):
        raise ValueError("entry mark must use locked entry_ts")
    if parse_timestamp(
        exit_mark.get("observed_at"), "exit.observed_at"
    ) != parse_timestamp(slot["exit_ts"], "exit_ts"):
        raise ValueError("exit mark must use locked exit_ts")
    long_symbol = expression["long_leg"]["contract_symbol"]
    short_symbol = expression["short_leg"]["contract_symbol"]
    for name, payload in (("entry", entry), ("exit", exit_mark)):
        if payload.get("long_contract_symbol") != long_symbol:
            raise ValueError(f"{name} long contract does not match expression")
        if payload.get("short_contract_symbol") != short_symbol:
            raise ValueError(f"{name} short contract does not match expression")

    entry_long_ask = _safe_float(
        entry.get("long_ask"), "entry.long_ask", allow_zero=False
    )
    entry_short_bid = _safe_float(entry.get("short_bid"), "entry.short_bid")
    exit_long_bid = _safe_float(exit_mark.get("long_bid"), "exit.long_bid")
    exit_short_ask = _safe_float(exit_mark.get("short_ask"), "exit.short_ask")
    entry_debit = round((entry_long_ask - entry_short_bid) * 100.0, 2)
    if entry_debit <= 0:
        raise ValueError("natural entry debit must be positive")
    if entry_debit > float(candidate["risk_cap_dollars"]):
        raise ValueError("natural entry debit exceeds candidate risk cap")
    exit_credit = round(max(0.0, exit_long_bid - exit_short_ask) * 100.0, 2)
    entry_ref = _source_ref(entry.get("source") or {}, "entry.source")
    exit_ref = _source_ref(exit_mark.get("source") or {}, "exit.source")
    return {
        "schema": PAPER_MARK_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": candidate["slot_id"],
        "entry_ts": slot["entry_ts"],
        "exit_ts": slot["exit_ts"],
        "label_available_ts": slot["label_available_ts"],
        "published_at": now.isoformat(),
        "vehicle": "debit_spread",
        "entry_method": manifest["fill_rules"]["debit_spread_entry"],
        "exit_method": manifest["fill_rules"]["debit_spread_exit"],
        "entry_debit_dollars": entry_debit,
        "exit_credit_dollars": exit_credit,
        "leg_count": 2,
        "source_refs": [
            f"mark-snapshot-sha256://{_artifact_sha256(snapshot)}",
            entry_ref,
            exit_ref,
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def build_publication(
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase5: dict[str, Any],
    *,
    now: datetime,
    manifest: dict[str, Any] | None = None,
    option_snapshot: dict[str, Any] | None = None,
    mark_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_phase5_chain(
        phase3,
        phase4,
        phase5,
        manifest=manifest,
        option_snapshot=option_snapshot,
    )
    state = phase5.get("state")
    if state == "upstream_not_ready":
        if mark_snapshot is not None:
            raise ValueError("upstream_not_ready must not consume paper mark evidence")
        return {
            **_base(phase5, now),
            "state": "upstream_not_ready",
            "evaluator_accounting": "none",
            "paper_mark": None,
            "evaluation_receipt": None,
        }
    if manifest is None:
        raise ValueError("mature receipt states require the locked manifest")
    verify_evaluator_source_lock(
        manifest, source_bundle_sha256(evaluator_source_paths())
    )

    if state == "abstained":
        if mark_snapshot is not None:
            raise ValueError("abstained review must not consume paper mark evidence")
        receipt = evaluate_candidate(manifest, _candidate(phase3, manifest), None)
        return {
            **_base(phase5, now),
            "state": "abstention_receipt_published",
            "evaluator_accounting": "stand_down",
            "paper_mark": None,
            "evaluation_receipt": receipt,
        }
    if state == "vetoed":
        if mark_snapshot is not None:
            raise ValueError("vetoed review must not consume paper mark evidence")
        reason = "; ".join(phase5.get("reasons") or ["Phase 5 veto"])
        receipt = _mark_failure_receipt(manifest, phase5, reason)
        return {
            **_base(phase5, now),
            "state": "rejection_receipt_published",
            "evaluator_accounting": "zero_utility_rejection",
            "paper_mark": None,
            "evaluation_receipt": receipt,
        }
    if state != "paper_expression_accepted" or phase5.get("verdict") != "accept":
        raise ValueError(f"unsupported Phase 5 state: {state}")
    if mark_snapshot is None:
        raise ValueError("accepted expression requires exact paper mark evidence")
    try:
        mark = _build_mark(manifest, phase3, phase4, mark_snapshot, now)
        receipt = evaluate_candidate(manifest, _candidate(phase3, manifest), mark)
    except (KeyError, TypeError, ValueError) as exc:
        receipt = _mark_failure_receipt(manifest, phase5, f"paper mark rejected: {exc}")
        return {
            **_base(phase5, now),
            "state": "mark_rejected",
            "evaluator_accounting": "zero_utility_rejection",
            "paper_mark": None,
            "evaluation_receipt": receipt,
        }
    return {
        **_base(phase5, now),
        "state": "evaluation_receipt_published",
        "evaluator_accounting": "evaluated",
        "paper_mark": mark,
        "evaluation_receipt": receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-publication", required=True, type=Path)
    parser.add_argument("--phase4-publication", required=True, type=Path)
    parser.add_argument("--phase5-review", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--option-snapshot", type=Path)
    parser.add_argument("--mark-snapshot", type=Path)
    parser.add_argument("--now")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit(
            "Paper Mark Publisher is artifact-only; --no-network is required"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only artifact: {args.output}")
    phase3 = _load_optional(args.phase3_publication)
    phase4 = _load_optional(args.phase4_publication)
    phase5 = _load_optional(args.phase5_review)
    manifest = _load_optional(args.manifest)
    option_snapshot = _load_optional(args.option_snapshot)
    mark_snapshot = _load_optional(args.mark_snapshot)
    now = parse_timestamp(args.now or phase5.get("reviewed_at"), "now")
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=now,
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=mark_snapshot,
    )
    args.output.write_text(canonical_json(publication) + "\n", encoding="utf-8")
    print(
        json.dumps({"state": publication["state"], "slot_id": publication["slot_id"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
