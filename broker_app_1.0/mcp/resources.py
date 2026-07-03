from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from runtime.argus_mcp_wrapper import (
    WrapperContext,
    get_execution_card,
    get_latest_state,
)

from .auth import CapabilityProfile
from .errors import CapabilityDeniedError, InvalidRequestError, UnknownResourceError

RESOURCE_NAMES = (
    "sharpedge://state/latest",
    "sharpedge://execution/card/latest",
    "sharpedge://permission/latest",
    "sharpedge://positions/latest",
    "sharpedge://handoff/latest",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json_resource(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_dict_request(request: dict[str, Any] | None) -> dict[str, Any]:
    if request is None:
        return {}
    if not isinstance(request, dict):
        raise InvalidRequestError("resource request must be a JSON object")
    return request


def _resource_response(
    *,
    resource: str,
    authority: str,
    source_refs: list[str],
    contents: Any,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "resource": resource,
        "authority": authority,
        "mutability": "read_only",
        "generated_at": _utc_now(),
        "source_refs": source_refs,
        "contents": contents,
    }


def _tool_payload_or_raise(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "ok":
        raise InvalidRequestError(
            payload.get("message") or f"wrapper returned status={payload.get('status')}"
        )
    return payload


def _state_latest(context: WrapperContext, _: dict[str, Any]) -> dict[str, Any]:
    payload = _tool_payload_or_raise(get_latest_state(context=context))
    return _resource_response(
        resource="sharpedge://state/latest",
        authority="SharpEdge",
        source_refs=payload["source_refs"],
        contents=payload["state"],
    )


def _execution_card_latest(
    context: WrapperContext, _: dict[str, Any]
) -> dict[str, Any]:
    payload = _tool_payload_or_raise(get_execution_card(context=context))
    return _resource_response(
        resource="sharpedge://execution/card/latest",
        authority="SharpEdge",
        source_refs=payload["source_refs"],
        contents=payload["execution_card"],
    )


def _permission_latest(context: WrapperContext, _: dict[str, Any]) -> dict[str, Any]:
    payload = _tool_payload_or_raise(get_execution_card(context=context))
    return _resource_response(
        resource="sharpedge://permission/latest",
        authority="SharpEdge",
        source_refs=payload["source_refs"],
        contents=payload["execution_card"],
    )


def _positions_latest(context: WrapperContext, _: dict[str, Any]) -> dict[str, Any]:
    path = context.outputs_dir / "robinhood_live_positions.json"
    return _resource_response(
        resource="sharpedge://positions/latest",
        authority="Robinhood Bridge",
        source_refs=[str(path)],
        contents=_read_json_resource(path),
    )


def _handoff_latest(context: WrapperContext, _: dict[str, Any]) -> dict[str, Any]:
    return _resource_response(
        resource="sharpedge://handoff/latest",
        authority="SharpEdge-Robinhood-Bridge",
        source_refs=[str(context.handoff_path)],
        contents=_read_json_resource(context.handoff_path),
    )


_RESOURCE_TABLE: dict[
    str, tuple[str, Callable[[WrapperContext, dict[str, Any]], dict[str, Any]]]
] = {
    "sharpedge://state/latest": ("read_state", _state_latest),
    "sharpedge://execution/card/latest": (
        "read_execution_card",
        _execution_card_latest,
    ),
    "sharpedge://permission/latest": ("read_permission", _permission_latest),
    "sharpedge://positions/latest": ("read_positions", _positions_latest),
    "sharpedge://handoff/latest": ("read_handoff", _handoff_latest),
}


def list_resources(
    capabilities: CapabilityProfile | None = None,
) -> list[dict[str, str]]:
    caps = capabilities or CapabilityProfile()
    items: list[dict[str, str]] = []
    for name, (capability, _) in _RESOURCE_TABLE.items():
        if caps.allows(capability):
            items.append({"resource": name, "required_capability": capability})
    return items


def read_resource(
    resource_name: str,
    request: dict[str, Any] | None = None,
    *,
    context: WrapperContext | None = None,
    capabilities: CapabilityProfile | None = None,
) -> dict[str, Any]:
    caps = capabilities or CapabilityProfile()
    ctx = context or WrapperContext.default()
    normalized_request = _require_dict_request(request)
    try:
        capability, handler = _RESOURCE_TABLE[resource_name]
    except KeyError as exc:
        raise UnknownResourceError(resource_name) from exc
    if not caps.allows(capability):
        raise CapabilityDeniedError(capability)
    return handler(ctx, normalized_request)
