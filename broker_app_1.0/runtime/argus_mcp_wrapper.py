from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOOL_NAMES = (
    "sharpedge.discover_surface",
    "sharpedge.get_latest_state",
    "sharpedge.get_execution_card",
    "sharpedge.explain_permission",
    "sharpedge.prepare_broker_handoff",
    "sharpedge.validate_handoff",
)
_HANDOFF_SCHEMA = "sharpedge.robinhood_execution_handoff.v1"
_DEFAULT_HANDOFF_NAME = "robinhood_execution_handoff.json"
_DEFAULT_SIGNAL_NAME = "signal.json"


@dataclass(frozen=True)
class WrapperContext:
    broker_app_root: Path
    sharpedge_root: Path
    bridge_root: Path
    outputs_dir: Path

    @classmethod
    def default(cls) -> "WrapperContext":
        broker_app_root = Path(__file__).resolve().parents[1]
        sharpedge_root = broker_app_root.parent
        return cls(
            broker_app_root=broker_app_root,
            sharpedge_root=sharpedge_root,
            bridge_root=sharpedge_root.parent / "SharpEdge-Robinhood-Bridge",
            outputs_dir=sharpedge_root / "outputs",
        )

    @property
    def signal_path(self) -> Path:
        return self.outputs_dir / _DEFAULT_SIGNAL_NAME

    @property
    def handoff_path(self) -> Path:
        return self.outputs_dir / _DEFAULT_HANDOFF_NAME


def _ctx(context: WrapperContext | None) -> WrapperContext:
    return context or WrapperContext.default()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_ref(*paths: Path | str) -> list[str]:
    return [str(path) for path in paths]


def _base_response(
    *,
    status: str,
    tool_name: str,
    authority: str,
    mutability: str,
    source_refs: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "tool_name": tool_name,
        "authority": authority,
        "mutability": mutability,
        "generated_at": _utc_now(),
        "source_refs": source_refs,
    }
    if extra:
        payload.update(extra)
    return payload


def _error_response(
    *,
    status: str,
    tool_name: str,
    authority: str,
    mutability: str,
    source_refs: list[str],
    error_code: str,
    message: str,
    retryable: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _base_response(
        status=status,
        tool_name=tool_name,
        authority=authority,
        mutability=mutability,
        source_refs=source_refs,
        extra=extra,
    )
    payload.update(
        {
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
        }
    )
    return payload


def _require_bool(name: str, value: Any) -> str | None:
    if not isinstance(value, bool):
        return f"{name} must be boolean."
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty JSON object in {path}")
    return data


def _load_surface_files(
    context: WrapperContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(
        context.broker_app_root / "manifests" / "argus_mcp_manifest.json"
    )
    inventory = _read_json(
        context.broker_app_root / "bridge" / "real_surface_inventory.json"
    )
    aliases = _read_json(context.broker_app_root / "tools" / "argus_tool_aliases.json")
    return manifest, inventory, aliases


def _bridge_exports(context: WrapperContext) -> dict[str, Any]:
    bridge_src = context.bridge_root / "src"
    if not bridge_src.exists():
        raise FileNotFoundError(bridge_src)
    bridge_src_text = str(bridge_src)
    if bridge_src_text not in sys.path:
        sys.path.insert(0, bridge_src_text)
    module = importlib.import_module("sharpedge_robinhood_bridge.cockpit_adapter")
    return {
        "plan_signal_handoff": getattr(module, "plan_signal_handoff"),
        "write_handoff_artifact": getattr(module, "write_handoff_artifact"),
    }


def _summarize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": signal.get("schema"),
        "symbol": signal.get("symbol"),
        "spot": signal.get("spot"),
        "gamma_regime": signal.get("gamma_regime"),
        "setup_tag": signal.get("setup_tag"),
        "trade_permission": signal.get("trade_permission") or {},
    }


def _load_signal(context: WrapperContext) -> dict[str, Any]:
    return _read_json(context.signal_path)


def _load_execution_card(signal: dict[str, Any]) -> dict[str, Any]:
    card = signal.get("trade_permission")
    if not isinstance(card, dict) or not card:
        raise ValueError(
            "Latest SharpEdge state does not include a usable trade_permission payload."
        )
    return dict(card)


def _listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _explanation_summary(
    *,
    score: Any,
    gate: Any,
    supporting: list[str],
    warnings: list[str],
    detail_level: str,
) -> str:
    prefix = f"Permission is {score} and gate is {gate}."
    if detail_level == "brief":
        return prefix
    if detail_level == "deep":
        parts = [prefix]
        if supporting:
            parts.append("Supporting reasons: " + "; ".join(supporting) + ".")
        if warnings:
            parts.append("Warnings: " + "; ".join(warnings) + ".")
        return " ".join(parts)
    notes: list[str] = []
    if supporting:
        notes.append(f"{len(supporting)} supporting reason(s) are present")
    if warnings:
        notes.append(f"{len(warnings)} warning reason(s) are present")
    return prefix if not notes else prefix + " " + "; ".join(notes) + "."


def discover_surface(
    *,
    include_resources: bool = True,
    include_tools: bool = True,
    include_authority_map: bool = True,
    include_legacy_aliases: bool = False,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    source_refs = _source_ref(
        ctx.broker_app_root / "bridge" / "real_surface_inventory.json",
        ctx.broker_app_root / "tools" / "argus_tool_aliases.json",
        ctx.broker_app_root / "docs" / "authority_map.md",
    )
    for name, value in {
        "include_resources": include_resources,
        "include_tools": include_tools,
        "include_authority_map": include_authority_map,
        "include_legacy_aliases": include_legacy_aliases,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.discover_surface",
                authority="Argus-MCP-Wrapper",
                mutability="read_only",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    try:
        manifest, inventory, aliases = _load_surface_files(ctx)
    except FileNotFoundError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.discover_surface",
            authority="Argus-MCP-Wrapper",
            mutability="read_only",
            source_refs=source_refs,
            error_code="surface_file_missing",
            message=f"Surface contract file missing: {exc}",
            retryable=False,
        )
    except ValueError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.discover_surface",
            authority="Argus-MCP-Wrapper",
            mutability="read_only",
            source_refs=source_refs,
            error_code="surface_file_invalid",
            message=str(exc),
            retryable=False,
        )

    payload: dict[str, Any] = {
        "surface": {
            "tools": inventory.get("tools") if include_tools else [],
            "resources": inventory.get("resources") if include_resources else [],
            "authority_boundary": (
                manifest.get("authority_boundary") if include_authority_map else {}
            )
            or {},
        }
    }
    if include_legacy_aliases:
        canonical = set(aliases.get("canonical_names") or [])
        alias_names = [
            str(entry.get("argus_tool"))
            for entry in aliases.get("aliases") or []
            if str(entry.get("argus_tool")) not in canonical
        ]
        payload["legacy_aliases"] = alias_names
    return _base_response(
        status="ok",
        tool_name="sharpedge.discover_surface",
        authority="Argus-MCP-Wrapper",
        mutability="read_only",
        source_refs=source_refs,
        extra=payload,
    )


def get_latest_state(
    *,
    source: str = "latest",
    include_artifact_path: bool = True,
    include_raw_signal: bool = True,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    source_refs = _source_ref(ctx.signal_path)
    if source != "latest":
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.get_latest_state",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="unsupported_source",
            message="source must be 'latest'.",
            retryable=False,
        )
    for name, value in {
        "include_artifact_path": include_artifact_path,
        "include_raw_signal": include_raw_signal,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.get_latest_state",
                authority="SharpEdge",
                mutability="read_only",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    try:
        signal = _load_signal(ctx)
    except FileNotFoundError:
        return _error_response(
            status="not_found",
            tool_name="sharpedge.get_latest_state",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="latest_state_missing",
            message=f"Latest SharpEdge state not found at {ctx.signal_path}.",
            retryable=True,
        )
    except ValueError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.get_latest_state",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="latest_state_invalid",
            message=str(exc),
            retryable=False,
        )
    payload: dict[str, Any] = {
        "state": signal if include_raw_signal else _summarize_signal(signal)
    }
    if include_artifact_path:
        payload["artifact_path"] = str(ctx.signal_path)
    return _base_response(
        status="ok",
        tool_name="sharpedge.get_latest_state",
        authority="SharpEdge",
        mutability="read_only",
        source_refs=source_refs,
        extra=payload,
    )


def get_execution_card(
    *,
    source: str = "latest",
    include_reasons: bool = True,
    include_execution_flow: bool = True,
    include_execution_hierarchy: bool = True,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    source_refs = _source_ref(
        ctx.signal_path,
        ctx.sharpedge_root / "cockpit" / "execution_card_builder.py",
    )
    if source != "latest":
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.get_execution_card",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="unsupported_source",
            message="source must be 'latest'.",
            retryable=False,
        )
    for name, value in {
        "include_reasons": include_reasons,
        "include_execution_flow": include_execution_flow,
        "include_execution_hierarchy": include_execution_hierarchy,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.get_execution_card",
                authority="SharpEdge",
                mutability="read_only",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    try:
        card = _load_execution_card(_load_signal(ctx))
    except FileNotFoundError:
        return _error_response(
            status="not_found",
            tool_name="sharpedge.get_execution_card",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="latest_state_missing",
            message=f"Latest SharpEdge state not found at {ctx.signal_path}.",
            retryable=True,
        )
    except ValueError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.get_execution_card",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="execution_card_unavailable",
            message=str(exc),
            retryable=False,
        )
    if not include_reasons:
        card.pop("supporting_reasons", None)
        card.pop("warning_reasons", None)
    if not include_execution_flow:
        card.pop("execution_flow", None)
    if not include_execution_hierarchy:
        card.pop("execution_hierarchy", None)
    return _base_response(
        status="ok",
        tool_name="sharpedge.get_execution_card",
        authority="SharpEdge",
        mutability="read_only",
        source_refs=source_refs,
        extra={"execution_card": card},
    )


def explain_permission(
    *,
    source: str = "latest",
    detail_level: str = "standard",
    audience: str = "operator",
    include_risk: bool = True,
    include_invalidation: bool = True,
    include_reasons: bool = True,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    source_refs = _source_ref(
        ctx.signal_path,
        ctx.sharpedge_root / "cockpit" / "execution_card_builder.py",
    )
    if source != "latest":
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.explain_permission",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="unsupported_source",
            message="source must be 'latest'.",
            retryable=False,
        )
    if detail_level not in {"brief", "standard", "deep"}:
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.explain_permission",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="unsupported_detail_level",
            message="detail_level must be one of: brief, standard, deep.",
            retryable=False,
        )
    if not isinstance(audience, str) or not audience.strip():
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.explain_permission",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="invalid_audience",
            message="audience must be a non-empty string.",
            retryable=False,
        )
    for name, value in {
        "include_risk": include_risk,
        "include_invalidation": include_invalidation,
        "include_reasons": include_reasons,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.explain_permission",
                authority="SharpEdge",
                mutability="read_only",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    try:
        signal = _load_signal(ctx)
        card = _load_execution_card(signal)
    except FileNotFoundError:
        return _error_response(
            status="not_found",
            tool_name="sharpedge.explain_permission",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="latest_state_missing",
            message=f"Latest SharpEdge state not found at {ctx.signal_path}.",
            retryable=True,
        )
    except ValueError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.explain_permission",
            authority="SharpEdge",
            mutability="read_only",
            source_refs=source_refs,
            error_code="execution_card_unavailable",
            message=str(exc),
            retryable=False,
        )

    supporting = _listify(card.get("supporting_reasons")) if include_reasons else []
    warnings = _listify(card.get("warning_reasons")) if include_reasons else []
    explanation = {
        "score": card.get(
            "trade_permission_score", card.get("execution_permission_score")
        ),
        "gate": card.get("trade_gate"),
        "plain_language_summary": _explanation_summary(
            score=card.get(
                "trade_permission_score", card.get("execution_permission_score")
            ),
            gate=card.get("trade_gate"),
            supporting=supporting,
            warnings=warnings,
            detail_level=detail_level,
        ),
        "supporting_reasons": supporting,
        "warning_reasons": warnings,
        "risk_notes": warnings if include_risk else [],
        "invalidation_notes": _listify(signal.get("invalidation"))
        if include_invalidation
        else [],
        "audience": audience,
    }
    return _base_response(
        status="ok",
        tool_name="sharpedge.explain_permission",
        authority="SharpEdge",
        mutability="read_only",
        source_refs=source_refs,
        extra={"explanation": explanation},
    )


def prepare_broker_handoff(
    *,
    source: str = "latest",
    command: str = "order_submit",
    operator_approved: bool = True,
    test: bool = False,
    write_latest_artifact: bool = True,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    source_refs = _source_ref(
        ctx.signal_path,
        ctx.bridge_root / "src" / "sharpedge_robinhood_bridge" / "cockpit_adapter.py",
        ctx.bridge_root / "src" / "sharpedge_robinhood_bridge" / "trade_intent.py",
    )
    if source != "latest":
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="unsupported_source",
            message="source must be 'latest'.",
            retryable=False,
        )
    if not isinstance(command, str) or not command.strip():
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="invalid_command",
            message="command must be a non-empty string.",
            retryable=False,
        )
    for name, value in {
        "operator_approved": operator_approved,
        "test": test,
        "write_latest_artifact": write_latest_artifact,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.prepare_broker_handoff",
                authority="SharpEdge-Robinhood-Bridge",
                mutability="write_artifact",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    if operator_approved is not True:
        return _error_response(
            status="blocked",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="operator_approval_required",
            message="operator_approved must be true before preparing a broker handoff.",
            retryable=False,
        )
    try:
        bridge = _bridge_exports(ctx)
        handoff = bridge["plan_signal_handoff"](
            ctx.signal_path, command=command, test=test
        )
    except FileNotFoundError as exc:
        return _error_response(
            status="not_found",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="latest_state_missing",
            message=str(exc),
            retryable=True,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _error_response(
            status="error",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="handoff_preparation_failed",
            message=str(exc),
            retryable=False,
        )

    command_plan = handoff.get("command_plan") or {}
    if command_plan.get("status") == "stand_down":
        return _error_response(
            status="blocked",
            tool_name="sharpedge.prepare_broker_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="write_artifact",
            source_refs=source_refs,
            error_code="stand_down",
            message="SharpEdge did not produce a broker-ready handoff for this state.",
            retryable=True,
            extra={"handoff": handoff},
        )

    artifact_path = ctx.handoff_path
    if write_latest_artifact:
        try:
            artifact_path = bridge["write_handoff_artifact"](
                handoff,
                out_dir=ctx.outputs_dir,
                latest_name=_DEFAULT_HANDOFF_NAME,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return _error_response(
                status="error",
                tool_name="sharpedge.prepare_broker_handoff",
                authority="SharpEdge-Robinhood-Bridge",
                mutability="write_artifact",
                source_refs=source_refs,
                error_code="handoff_write_failed",
                message=str(exc),
                retryable=False,
                extra={"handoff": handoff},
            )
    return _base_response(
        status="ok",
        tool_name="sharpedge.prepare_broker_handoff",
        authority="SharpEdge-Robinhood-Bridge",
        mutability="write_artifact",
        source_refs=source_refs,
        extra={"handoff": handoff, "artifact_path": str(artifact_path)},
    )


def validate_handoff(
    *,
    handoff_path: str = "",
    use_latest_if_missing: bool = True,
    check_route: bool = True,
    check_approval_policy: bool = True,
    check_payload_contracts: bool = True,
    context: WrapperContext | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    resolved_path = (
        Path(handoff_path).expanduser() if handoff_path.strip() else ctx.handoff_path
    )
    source_refs = _source_ref(
        resolved_path,
        ctx.bridge_root / "src" / "sharpedge_robinhood_bridge" / "catalog.py",
        ctx.bridge_root / "src" / "sharpedge_robinhood_bridge" / "payload_contracts.py",
    )
    for name, value in {
        "use_latest_if_missing": use_latest_if_missing,
        "check_route": check_route,
        "check_approval_policy": check_approval_policy,
        "check_payload_contracts": check_payload_contracts,
    }.items():
        error = _require_bool(name, value)
        if error:
            return _error_response(
                status="invalid_input",
                tool_name="sharpedge.validate_handoff",
                authority="SharpEdge-Robinhood-Bridge",
                mutability="validate_only",
                source_refs=source_refs,
                error_code="invalid_boolean_flag",
                message=error,
                retryable=False,
            )
    if not handoff_path.strip() and not use_latest_if_missing:
        return _error_response(
            status="invalid_input",
            tool_name="sharpedge.validate_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="validate_only",
            source_refs=source_refs,
            error_code="missing_handoff_reference",
            message="Provide handoff_path or set use_latest_if_missing to true.",
            retryable=False,
        )
    try:
        handoff = _read_json(resolved_path)
    except FileNotFoundError:
        return _error_response(
            status="not_found",
            tool_name="sharpedge.validate_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="validate_only",
            source_refs=source_refs,
            error_code="handoff_missing",
            message=f"Handoff not found at {resolved_path}.",
            retryable=True,
        )
    except ValueError as exc:
        return _error_response(
            status="error",
            tool_name="sharpedge.validate_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="validate_only",
            source_refs=source_refs,
            error_code="handoff_invalid",
            message=str(exc),
            retryable=False,
        )

    command_plan = handoff.get("command_plan") or {}
    delegation = handoff.get("delegation") or {}
    broker_payload = delegation.get("broker_payload") or {}
    payload_contracts = broker_payload.get("payload_contracts") or {}
    operator_gate = handoff.get("operator_gate") or {}
    route = str(command_plan.get("route") or "unknown")
    approval_policy = str(command_plan.get("approval_policy") or "unknown")
    plan_status = str(command_plan.get("status") or "unknown")

    issues: list[str] = []
    warnings: list[str] = []
    if handoff.get("schema") != _HANDOFF_SCHEMA:
        issues.append(f"Expected schema '{_HANDOFF_SCHEMA}'.")
    if check_route and route != "chatgpt_delegate":
        issues.append(f"Route '{route}' is not delegation-ready.")
    if check_approval_policy and approval_policy != "operator_confirm_required":
        issues.append(
            f"Approval policy '{approval_policy}' does not satisfy operator confirmation requirements."
        )
    if check_payload_contracts and not isinstance(payload_contracts, dict):
        issues.append("payload_contracts must be a JSON object.")
    if (
        check_payload_contracts
        and isinstance(payload_contracts, dict)
        and not payload_contracts
    ):
        issues.append("payload_contracts are missing from the delegation payload.")
    if operator_gate.get("required") is not True:
        issues.append(
            "operator_gate.required must be true for delegation-ready handoffs."
        )
    if plan_status != "awaiting_operator_confirm":
        warnings.append(f"command_plan.status is '{plan_status}'.")

    validation = {
        "valid": not issues,
        "route": route,
        "approval_policy": approval_policy,
        "ready_for_delegation": not issues,
        "issues": issues,
        "warnings": warnings,
    }
    if issues:
        return _error_response(
            status="blocked",
            tool_name="sharpedge.validate_handoff",
            authority="SharpEdge-Robinhood-Bridge",
            mutability="validate_only",
            source_refs=source_refs,
            error_code="handoff_not_ready",
            message="Handoff failed validation checks.",
            retryable=False,
            extra={"validation": validation},
        )
    return _base_response(
        status="ok",
        tool_name="sharpedge.validate_handoff",
        authority="SharpEdge-Robinhood-Bridge",
        mutability="validate_only",
        source_refs=source_refs,
        extra={"validation": validation},
    )
