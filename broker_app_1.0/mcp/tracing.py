from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Any

from runtime.argus_mcp_wrapper import WrapperContext

TRACE_FILE_NAME = "argus_mcp_trace.jsonl"
TRACE_SCHEMA = "argus.mcp_trace.v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def start_timer() -> float:
    return perf_counter()


def duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def build_request_id() -> str:
    return f"argus-{uuid4().hex[:12]}"


def trace_path_for_context(context: WrapperContext) -> Path:
    return context.outputs_dir / TRACE_FILE_NAME


def enrich_response(
    payload: dict[str, Any],
    *,
    request_id: str,
    api_version: int,
    duration_ms_value: float,
    trace_path: Path,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["request_id"] = request_id
    enriched["api_version"] = api_version
    enriched["duration_ms"] = duration_ms_value
    enriched["trace_artifact_path"] = str(trace_path)
    return enriched


def build_trace_event(
    *,
    request_id: str,
    api_version: int,
    target_kind: str,
    target_name: str,
    wrapper_name: str,
    payload: dict[str, Any],
    duration_ms_value: float,
) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "generated_at": utc_now(),
        "request_id": request_id,
        "api_version": api_version,
        "transport": "in_process_skeleton",
        "target_kind": target_kind,
        "target_name": target_name,
        "wrapper": wrapper_name,
        "status": payload.get("status", "unknown"),
        "authority": payload.get("authority", "unknown"),
        "duration_ms": duration_ms_value,
        "source_refs": list(payload.get("source_refs") or []),
        "error_code": payload.get("error_code", ""),
    }


def append_trace_event(context: WrapperContext, event: dict[str, Any]) -> Path:
    trace_path = trace_path_for_context(context)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return trace_path
