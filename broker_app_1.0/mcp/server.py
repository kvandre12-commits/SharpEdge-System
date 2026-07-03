from __future__ import annotations

from typing import Any

from runtime.argus_mcp_wrapper import WrapperContext

from .auth import CapabilityProfile, default_capabilities
from .errors import MCPError
from .resources import list_resources, read_resource
from .tools import call_tool, list_tools

API_VERSION = 1


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

    def call_tool(
        self, tool_name: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            return call_tool(
                tool_name,
                request,
                context=self.context,
                capabilities=self.capabilities,
            )
        except MCPError as exc:
            return exc.to_response()

    def read_resource(
        self,
        resource_name: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return read_resource(
                resource_name,
                request,
                context=self.context,
                capabilities=self.capabilities,
            )
        except MCPError as exc:
            return exc.to_response()
