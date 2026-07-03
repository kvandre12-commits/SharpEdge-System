from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPError(Exception):
    code: str
    message: str
    retryable: bool = False
    extra: dict[str, Any] | None = None

    def to_response(self) -> dict[str, Any]:
        payload = {
            "status": "error",
            "error_code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


class UnknownToolError(MCPError):
    def __init__(self, tool_name: str):
        super().__init__(
            code="unknown_tool",
            message=f"Unknown MCP tool: {tool_name}",
            retryable=False,
        )


class UnknownResourceError(MCPError):
    def __init__(self, resource_name: str):
        super().__init__(
            code="unknown_resource",
            message=f"Unknown MCP resource: {resource_name}",
            retryable=False,
        )


class CapabilityDeniedError(MCPError):
    def __init__(self, capability: str):
        super().__init__(
            code="capability_denied",
            message=f"Capability '{capability}' is not enabled.",
            retryable=False,
            extra={"required_capability": capability},
        )


class InvalidRequestError(MCPError):
    def __init__(self, message: str):
        super().__init__(
            code="invalid_request",
            message=message,
            retryable=False,
        )
