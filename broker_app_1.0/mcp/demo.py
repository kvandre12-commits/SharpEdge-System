from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runtime.argus_mcp_wrapper import WrapperContext

from .auth import default_capabilities
from .server import ArgusMCPServer

API_VERSION = 1
_DEFAULT_ARTIFACT_NAME = "argus_mcp_execution_card_demo.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_execution_card_demo(
    *,
    context: WrapperContext | None = None,
    detail_level: str = "standard",
    write_artifact: bool = True,
    artifact_name: str = _DEFAULT_ARTIFACT_NAME,
) -> dict[str, Any]:
    ctx = context or WrapperContext.default()
    server = ArgusMCPServer(context=ctx, capabilities=default_capabilities())
    card_response = server.call_tool("sharpedge.get_execution_card")
    explanation_response = server.call_tool(
        "sharpedge.explain_permission",
        {"detail_level": detail_level},
    )

    payload = {
        "schema": "argus.mcp_demo.execution_card.v1",
        "api_version": API_VERSION,
        "generated_at": _utc_now(),
        "server": server.describe(),
        "flow": {
            "tool": "sharpedge.get_execution_card",
            "explanation_tool": "sharpedge.explain_permission",
            "resource": "sharpedge://execution/card/latest",
            "signal_path": str(ctx.signal_path),
        },
        "execution_card_response": card_response,
        "explanation_response": explanation_response,
    }
    if write_artifact:
        target = ctx.outputs_dir / artifact_name
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["artifact_path"] = str(target)
    return payload


def main() -> None:
    payload = run_execution_card_demo()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
