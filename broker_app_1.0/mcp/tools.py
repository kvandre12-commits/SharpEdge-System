from __future__ import annotations

from typing import Any, Callable

from runtime.argus_mcp_wrapper import (
    TOOL_NAMES,
    WrapperContext,
    discover_surface,
    explain_permission,
    get_execution_card,
    get_latest_state,
    prepare_broker_handoff,
    validate_handoff,
)

from .auth import CapabilityProfile
from .errors import CapabilityDeniedError, InvalidRequestError, UnknownToolError


def _require_dict_request(request: dict[str, Any] | None) -> dict[str, Any]:
    if request is None:
        return {}
    if not isinstance(request, dict):
        raise InvalidRequestError("tool request must be a JSON object")
    return request


_TOOL_TABLE: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "sharpedge.discover_surface": ("read_state", discover_surface),
    "sharpedge.get_latest_state": ("read_state", get_latest_state),
    "sharpedge.get_execution_card": ("read_execution_card", get_execution_card),
    "sharpedge.explain_permission": ("read_permission", explain_permission),
    "sharpedge.prepare_broker_handoff": ("prepare_handoff", prepare_broker_handoff),
    "sharpedge.validate_handoff": ("validate_handoff", validate_handoff),
}


def list_tools(capabilities: CapabilityProfile | None = None) -> list[dict[str, str]]:
    caps = capabilities or CapabilityProfile()
    items: list[dict[str, str]] = []
    for name in TOOL_NAMES:
        capability, _ = _TOOL_TABLE[name]
        if caps.allows(capability):
            items.append({"tool": name, "required_capability": capability})
    return items


def call_tool(
    tool_name: str,
    request: dict[str, Any] | None = None,
    *,
    context: WrapperContext | None = None,
    capabilities: CapabilityProfile | None = None,
) -> dict[str, Any]:
    caps = capabilities or CapabilityProfile()
    ctx = context or WrapperContext.default()
    normalized_request = _require_dict_request(request)
    try:
        capability, handler = _TOOL_TABLE[tool_name]
    except KeyError as exc:
        raise UnknownToolError(tool_name) from exc
    if not caps.allows(capability):
        raise CapabilityDeniedError(capability)
    return handler(context=ctx, **normalized_request)
