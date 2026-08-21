from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from scripts.alpha_swarm.contracts import (
    RECEIPT_SCHEMA,
    SCORE_SCHEMA,
    ContractError,
    manifest_sha256,
    payload_sha256,
    slots_by_id,
    validate_candidate,
    validate_manifest,
    validate_mark,
)


def _positive_float(value: Any, field: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be numeric") from exc
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{field} must be {qualifier}")
    return number


def _equity_pnl(
    manifest: dict[str, Any], candidate: dict[str, Any], mark: dict[str, Any]
) -> tuple[float, float]:
    fills = manifest.get("fill_rules") or {}
    if mark.get("entry_method") != fills.get("equity_entry"):
        raise ContractError("equity entry method does not match the locked fill rule")
    if mark.get("exit_method") != fills.get("equity_exit"):
        raise ContractError("equity exit method does not match the locked fill rule")
    entry = _positive_float(mark.get("entry_price"), "entry_price")
    exit_price = _positive_float(mark.get("exit_price"), "exit_price")
    quantity = _positive_float(mark.get("quantity"), "quantity")
    sign = 1.0 if candidate["decision"] == "long" else -1.0
    gross = (exit_price - entry) * quantity * sign
    per_side_bps = _positive_float(
        (manifest.get("cost_model") or {}).get("equity_per_side_bps"),
        "equity_per_side_bps",
        allow_zero=True,
    )
    costs = (entry + exit_price) * quantity * per_side_bps / 10000.0
    return gross - costs, costs


def _debit_spread_pnl(
    manifest: dict[str, Any], mark: dict[str, Any]
) -> tuple[float, float, float]:
    fills = manifest.get("fill_rules") or {}
    if mark.get("entry_method") != fills.get("debit_spread_entry"):
        raise ContractError(
            "debit-spread entry method does not match the locked fill rule"
        )
    if mark.get("exit_method") != fills.get("debit_spread_exit"):
        raise ContractError(
            "debit-spread exit method does not match the locked fill rule"
        )
    entry_debit = _positive_float(
        mark.get("entry_debit_dollars"), "entry_debit_dollars"
    )
    exit_credit = _positive_float(
        mark.get("exit_credit_dollars"), "exit_credit_dollars", allow_zero=True
    )
    leg_count = int(_positive_float(mark.get("leg_count"), "leg_count"))
    if leg_count != 2:
        raise ContractError("Phase 1 permits two-leg debit spreads only")
    per_leg_side = _positive_float(
        (manifest.get("cost_model") or {}).get("option_per_leg_per_side_dollars"),
        "option_per_leg_per_side_dollars",
        allow_zero=True,
    )
    costs = leg_count * 2 * per_leg_side
    return exit_credit - entry_debit - costs, entry_debit, costs


def evaluate_candidate(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    mark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slot = validate_candidate(manifest, candidate)
    locked_hash = manifest_sha256(manifest)
    if candidate["decision"] == "stand_down":
        if mark is not None:
            raise ContractError("stand_down candidates must not include a paper mark")
        net_pnl = 0.0
        max_loss = None
        costs = 0.0
        utility = 0.0
        status = "abstained"
    else:
        if mark is None:
            raise ContractError("directional candidates require a paper mark")
        validate_mark(manifest, candidate, mark)
        max_loss = _positive_float(
            candidate.get("risk_cap_dollars"), "risk_cap_dollars"
        )
        if mark["vehicle"] == "equity":
            net_pnl, costs = _equity_pnl(manifest, candidate, mark)
        else:
            net_pnl, actual_max_loss, costs = _debit_spread_pnl(manifest, mark)
            if actual_max_loss > max_loss:
                raise ContractError(
                    "debit-spread entry debit exceeds predeclared risk cap"
                )
        utility = max(-1.0, min(1.0, net_pnl / max_loss))
        status = "evaluated"

    return {
        "schema": RECEIPT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": locked_hash,
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "slot_id": candidate["slot_id"],
        "session_date": slot["session_date"],
        "symbol": candidate["symbol"],
        "candidate_sha256": payload_sha256(candidate),
        "status": status,
        "decision": candidate["decision"],
        "vehicle": mark.get("vehicle") if mark else None,
        "net_pnl_dollars": round(net_pnl, 4),
        "costs_dollars": round(costs, 4),
        "max_loss_dollars": round(max_loss, 4) if max_loss is not None else None,
        "utility": round(utility, 8),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "label_available_ts": slot["label_available_ts"],
    }


def rejection_receipt(
    manifest: dict[str, Any], slot_id: str, reason: str
) -> dict[str, Any]:
    slots = slots_by_id(manifest)
    if slot_id not in slots or not slots[slot_id]["eligible"]:
        raise ContractError("rejection receipt requires an eligible locked slot")
    if not reason.strip():
        raise ContractError("rejection reason is required")
    slot = slots[slot_id]
    return {
        "schema": RECEIPT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "slot_id": slot_id,
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "candidate_sha256": None,
        "status": "rejected",
        "decision": None,
        "vehicle": None,
        "net_pnl_dollars": 0.0,
        "costs_dollars": 0.0,
        "max_loss_dollars": None,
        "utility": 0.0,
        "reason": reason,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "label_available_ts": slot["label_available_ts"],
    }


def _validate_receipt(manifest: dict[str, Any], receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ContractError("receipt schema is invalid")
    if receipt.get("run_id") != manifest.get("run_id"):
        raise ContractError("receipt run_id does not match manifest")
    if receipt.get("manifest_sha256") != manifest_sha256(manifest):
        raise ContractError("receipt manifest hash does not match")
    if receipt.get("evaluator_source_sha256") != manifest.get(
        "evaluator_source_sha256"
    ):
        raise ContractError("receipt evaluator source hash does not match")
    if (
        receipt.get("paper_only") is not True
        or receipt.get("execution_permitted") is not False
    ):
        raise ContractError("receipt must remain paper-only and non-executable")
    utility = float(receipt.get("utility"))
    if not -1.0 <= utility <= 1.0:
        raise ContractError("receipt utility must remain clipped to [-1, 1]")


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def score_receipts(
    manifest: dict[str, Any], receipts: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    validate_manifest(manifest)
    slots = slots_by_id(manifest)
    eligible = {slot_id: slot for slot_id, slot in slots.items() if slot["eligible"]}
    receipt_by_slot: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        _validate_receipt(manifest, receipt)
        slot_id = str(receipt.get("slot_id") or "")
        if slot_id not in eligible:
            raise ContractError("receipt targets an unknown or ineligible slot")
        if slot_id in receipt_by_slot:
            raise ContractError("only one receipt is allowed per eligible slot")
        receipt_by_slot[slot_id] = receipt

    utility_by_slot = {
        slot_id: float(receipt_by_slot.get(slot_id, {}).get("utility", 0.0))
        for slot_id in eligible
    }
    by_session: dict[str, list[float]] = defaultdict(list)
    for slot_id, utility in utility_by_slot.items():
        by_session[str(eligible[slot_id]["session_date"])].append(utility)
    session_dates = sorted(by_session)
    if not session_dates:
        raise ContractError("manifest has no eligible sessions")

    metric = manifest["metric"]
    rng = random.Random(int(metric["bootstrap_seed"]))
    bootstrap_means: list[float] = []
    iterations = int(metric["bootstrap_iterations"])
    for _ in range(iterations):
        sampled_sessions = [rng.choice(session_dates) for _ in session_dates]
        sample = [
            utility for session in sampled_sessions for utility in by_session[session]
        ]
        bootstrap_means.append(mean(sample))

    utilities = list(utility_by_slot.values())
    lower_quantile = float(metric["lower_quantile"])
    return {
        "schema": SCORE_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "metric": metric["name"],
        "eligible_slot_count": len(eligible),
        "evaluated_receipt_count": len(receipt_by_slot),
        "missing_slot_count": len(eligible) - len(receipt_by_slot),
        "rejected_slot_count": sum(
            receipt.get("status") == "rejected" for receipt in receipt_by_slot.values()
        ),
        "observed_mean_utility": round(mean(utilities), 8),
        "lower_confidence_utility": round(
            _quantile(bootstrap_means, lower_quantile), 8
        ),
        "lower_quantile": lower_quantile,
        "bootstrap_iterations": iterations,
        "positive_lower_bound": _quantile(bootstrap_means, lower_quantile) > 0,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def verify_evaluator_source_lock(
    manifest: dict[str, Any], current_source_sha256: str
) -> None:
    validate_manifest(manifest)
    if current_source_sha256 != manifest["evaluator_source_sha256"]:
        raise ContractError("evaluator source changed after the manifest was locked")


__all__ = [
    "evaluate_candidate",
    "rejection_receipt",
    "score_receipts",
    "verify_evaluator_source_lock",
]
