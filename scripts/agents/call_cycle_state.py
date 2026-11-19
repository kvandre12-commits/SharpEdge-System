"""Pure buyer-to-seller lifecycle for a wallet-bounded long-call cycle."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

WALLET_CAP_DOLLARS = 250.0
STATES = frozenset(
    {
        "IDLE",
        "ENTRY_PROPOSED",
        "OPEN_POSITION",
        "SELLER_MANAGING",
        "EXIT_PENDING_CONFIRM",
        "CLOSED",
        "COOLDOWN",
    }
)


class CycleTransitionError(ValueError):
    """Raised when an event would violate buyer/seller ownership boundaries."""


def initial_state(symbol: str) -> dict[str, Any]:
    return {
        "schema": "sharpedge.call_cycle_state.v1",
        "symbol": symbol.upper(),
        "state": "IDLE",
        "wallet_cap_dollars": WALLET_CAP_DOLLARS,
        "reserved_dollars": 0.0,
        "position": None,
        "trigger_id": None,
        "operator_confirmation_required": True,
        "broker_action_allowed": False,
    }


def _amount(value: Any, field: str) -> float:
    try:
        amount = round(float(value), 2)
    except (TypeError, ValueError) as exc:
        raise CycleTransitionError(f"{field} must be numeric") from exc
    if amount < 0:
        raise CycleTransitionError(f"{field} cannot be negative")
    return amount


def transition(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    current = str(state.get("state") or "")
    if current not in STATES:
        raise CycleTransitionError(f"unknown state: {current}")
    event_type = str(event.get("type") or "")
    result = deepcopy(state)

    if event_type == "lead_approved":
        if current != "IDLE":
            raise CycleTransitionError("lead approval requires IDLE state")
        debit = _amount(event.get("max_debit_dollars"), "max_debit_dollars")
        if debit <= 0 or debit > WALLET_CAP_DOLLARS:
            raise CycleTransitionError("lead debit exceeds the $250 wallet")
        trigger_id = str(event.get("trigger_id") or "")
        if not trigger_id:
            raise CycleTransitionError("trigger_id is required")
        result.update(
            {
                "state": "ENTRY_PROPOSED",
                "reserved_dollars": debit,
                "trigger_id": trigger_id,
                "lead_id": event.get("lead_id"),
                "proposed_ladder": list(event.get("ladder") or []),
            }
        )
        return result

    if event_type == "entry_rejected":
        if current != "ENTRY_PROPOSED":
            raise CycleTransitionError("entry rejection requires ENTRY_PROPOSED")
        result.update({"state": "COOLDOWN", "reserved_dollars": 0.0})
        return result

    if event_type == "verified_fill":
        if current != "ENTRY_PROPOSED":
            raise CycleTransitionError("verified fill requires ENTRY_PROPOSED")
        if event.get("verified") is not True:
            raise CycleTransitionError("fill receipt must be explicitly verified")
        debit = _amount(event.get("filled_debit_dollars"), "filled_debit_dollars")
        reserved = _amount(result.get("reserved_dollars"), "reserved_dollars")
        if debit <= 0 or debit > min(reserved, WALLET_CAP_DOLLARS):
            raise CycleTransitionError("filled debit exceeds reserved wallet cash")
        contracts = list(event.get("contracts") or [])
        if not contracts:
            raise CycleTransitionError("verified fill requires owned contracts")
        result.update(
            {
                "state": "OPEN_POSITION",
                "reserved_dollars": debit,
                "position": {
                    "contracts": contracts,
                    "entry_debit_dollars": debit,
                    "fill_receipt_id": event.get("receipt_id"),
                    "profit_management_started": False,
                    "latest_return_pct": 0.0,
                },
            }
        )
        return result

    if event_type == "position_mark":
        if current not in {"OPEN_POSITION", "SELLER_MANAGING"}:
            raise CycleTransitionError("position mark requires an open owned position")
        return_pct = float(event.get("return_pct"))
        position = dict(result.get("position") or {})
        position["latest_return_pct"] = round(return_pct, 2)
        if return_pct >= 20:
            position["profit_management_started"] = True
            position["management_band"] = (
                "strong_take_profit_review"
                if return_pct >= 55
                else "tighten_exit_review"
                if return_pct >= 35
                else "protect_and_review"
            )
            result["state"] = "SELLER_MANAGING"
        result["position"] = position
        return result

    if event_type == "exit_proposed":
        if current != "SELLER_MANAGING":
            raise CycleTransitionError("exit proposal requires SELLER_MANAGING")
        result["state"] = "EXIT_PENDING_CONFIRM"
        result["exit_reason"] = event.get("reason")
        return result

    if event_type == "verified_close":
        if current != "EXIT_PENDING_CONFIRM":
            raise CycleTransitionError("verified close requires EXIT_PENDING_CONFIRM")
        if event.get("verified") is not True:
            raise CycleTransitionError("close receipt must be explicitly verified")
        proceeds = _amount(event.get("proceeds_dollars"), "proceeds_dollars")
        entry = _amount((result.get("position") or {}).get("entry_debit_dollars"), "entry_debit_dollars")
        result.update(
            {
                "state": "CLOSED",
                "reserved_dollars": 0.0,
                "close_receipt_id": event.get("receipt_id"),
                "realized_proceeds_dollars": proceeds,
                "realized_pnl_dollars": round(proceeds - entry, 2),
            }
        )
        return result

    if event_type == "begin_cooldown":
        if current != "CLOSED":
            raise CycleTransitionError("cooldown requires CLOSED")
        result["state"] = "COOLDOWN"
        return result

    if event_type == "trigger_reset":
        if current != "COOLDOWN":
            raise CycleTransitionError("trigger reset requires COOLDOWN")
        reset = initial_state(str(result.get("symbol") or ""))
        reset["previous_trigger_id"] = result.get("trigger_id")
        return reset

    raise CycleTransitionError(f"unsupported event {event_type!r} from {current}")


__all__ = ["CycleTransitionError", "initial_state", "transition"]
