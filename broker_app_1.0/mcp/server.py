from __future__ import annotations

from typing import Any

from runtime.argus_mcp_wrapper import WrapperContext

from .auth import CapabilityProfile, default_capabilities
from .errors import MCPError
from .resources import list_resources, read_resource
from .tools import call_tool, list_tools
from .tracing import (
    append_trace_event,
    build_request_id,
    build_trace_event,
    duration_ms,
    enrich_response,
    start_timer,
)

API_VERSION = 1

_TOOL_WRAPPER_NAMES = {
    "sharpedge.discover_surface": "discover_surface",
    "sharpedge.get_latest_state": "get_latest_state",
    "sharpedge.get_execution_card": "get_execution_card",
    "sharpedge.explain_permission": "explain_permission",
    "sharpedge.prepare_broker_handoff": "prepare_broker_handoff",
    "sharpedge.validate_handoff": "validate_handoff",
}
_RESOURCE_WRAPPER_NAMES = {
    "sharpedge://state/latest": "get_latest_state",
    "sharpedge://execution/card/latest": "get_execution_card",
    "sharpedge://permission/latest": "get_execution_card",
    "sharpedge://positions/latest": "read_json_resource",
    "sharpedge://handoff/latest": "read_json_resource",
}


class ArgusMCPServer:
    """Thin MCP-server skeleton that delegates to wrapper functions.

    This class is transport-agnostic on purpose. It models the server surface
    before any specific MCP SDK wiring is added.
    """

    def __init__(
        self,
        *,
        context: WrapperContext | None = None,
        capabilities: CapabilityProfile | None = None,
    ) -> None:
        self.context = context or WrapperContext.default()
        self.capabilities = capabilities or default_capabilities()

    def capability_snapshot(self) -> dict[str, bool]:
        return self.capabilities.as_dict()

    def describe(self) -> dict[str, Any]:
        return {
            "server_name": "argus-mcp-skeleton",
            "api_version": API_VERSION,
            "transport": "in_process_skeleton",
            "capabilities": self.capability_snapshot(),
            "tools": list_tools(self.capabilities),
            "resources": list_resources(self.capabilities),
        }

    def _finalize_response(
        self,
        *,
        payload: dict[str, Any],
        request_id: str,
        target_kind: str,
        target_name: str,
        wrapper_name: str,
        duration_ms_value: float,
    ) -> dict[str, Any]:
        event = build_trace_event(
            request_id=request_id,
            api_version=API_VERSION,
            target_kind=target_kind,
            target_name=target_name,
            wrapper_name=wrapper_name,
            payload=payload,
            duration_ms_value=duration_ms_value,
        )
        trace_path = append_trace_event(self.context, event)
        return enrich_response(
            payload,
            request_id=request_id,
            api_version=API_VERSION,
            duration_ms_value=duration_ms_value,
            trace_path=trace_path,
        )

    def call_tool(
        self, tool_name: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request_id = build_request_id()
        started_at = start_timer()
        try:
            payload = call_tool(
                tool_name,
                request,
                context=self.context,
                capabilities=self.capabilities,
            )
        except MCPError as exc:
            payload = exc.to_response()
        return self._finalize_response(
            payload=payload,
            request_id=request_id,
            target_kind="tool",
            target_name=tool_name,
            wrapper_name=_TOOL_WRAPPER_NAMES.get(tool_name, "unknown_wrapper"),
            duration_ms_value=duration_ms(started_at),
        )

    def read_resource(
        self,
        resource_name: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = build_request_id()
        started_at = start_timer()
        try:
            payload = read_resource(
                resource_name,
                request,
                context=self.context,
                capabilities=self.capabilities,
            )
        except MCPError as exc:
            payload = exc.to_response()
        return self._finalize_response(
            payload=payload,
            request_id=request_id,
            target_kind="resource",
            target_name=resource_name,
            wrapper_name=_RESOURCE_WRAPPER_NAMES.get(resource_name, "unknown_wrapper"),
            duration_ms_value=duration_ms(started_at),
        )
