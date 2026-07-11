#!/usr/bin/env python3
"""Build a cross-repo SharpEdge endpoint audit report."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SYSTEM_ROOT.parent
BRIDGE_ROOT = WORKSPACE_ROOT / "SharpEdge-Robinhood-Bridge"
ANDROID_ROOT = WORKSPACE_ROOT / "SharpEdge-Android"

ENDPOINT_AUDIT_JSON = SYSTEM_ROOT / "outputs/endpoint_audit_latest.json"
OUT_JSON = SYSTEM_ROOT / "outputs/workspace_endpoint_audit_latest.json"
OUT_MD = SYSTEM_ROOT / "outputs/workspace_endpoint_audit_latest.md"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_system_endpoint_audit() -> dict[str, Any]:
    if not ENDPOINT_AUDIT_JSON.exists():
        raise FileNotFoundError(
            "Run scripts/audit_endpoint_freshness.py first so workspace audit has live probe data."
        )
    return read_json(ENDPOINT_AUDIT_JSON)


def summarize_system_surfaces(endpoint_audit: dict[str, Any]) -> list[dict[str, Any]]:
    probes = endpoint_audit.get("probes") or []
    surfaces = []
    for probe in probes:
        surfaces.append(
            {
                "name": probe["name"],
                "status": probe["status"],
                "endpoint": probe.get("endpoint"),
                "freshness_signal": probe.get("last_bar_utc")
                or probe.get("latest_option_trade_time_raw")
                or probe.get("latest_observation_date")
                or probe.get("current_last_reported_date"),
                "notes": [],
            }
        )

    webhook_configured = bool(os.getenv("DISCORD_WEBHOOK_URL", "").strip())
    surfaces.extend(
        [
            {
                "name": "discord_trade_card_webhook",
                "status": "configured" if webhook_configured else "needs_env",
                "endpoint": "DISCORD_WEBHOOK_URL",
                "freshness_signal": None,
                "notes": [
                    "Write-only notification surface.",
                    "Fresh market-data semantics do not apply.",
                    "send_trade_card_to_discord.py now defines and validates the webhook env var.",
                ],
            },
            {
                "name": "discord_surface_execution_webhook",
                "status": "configured" if webhook_configured else "needs_env",
                "endpoint": "DISCORD_WEBHOOK_URL",
                "freshness_signal": None,
                "notes": [
                    "Write-only notification surface.",
                    "Posts outputs/surface_execution_card.* summaries to Discord.",
                ],
            },
        ]
    )
    return surfaces


def summarize_bridge_surfaces() -> dict[str, Any]:
    sys.path.insert(0, str(BRIDGE_ROOT / "src"))
    from sharpedge_robinhood_bridge.catalog import COMMAND_SPECS  # noqa: PLC0415

    route_counts = Counter(spec.route for spec in COMMAND_SPECS)
    public_reads = [
        spec.name for spec in COMMAND_SPECS if spec.route == "public_mcp_read"
    ]
    delegated_writes = [
        spec.name for spec in COMMAND_SPECS if spec.route == "chatgpt_delegate"
    ]
    local_logic = [
        spec.name for spec in COMMAND_SPECS if spec.route == "custom_logic_local"
    ]
    candidates = [
        spec.name for spec in COMMAND_SPECS if spec.route == "custom_logic_required"
    ]
    return {
        "route_counts": dict(route_counts),
        "public_read_commands": public_reads,
        "delegated_write_commands": delegated_writes,
        "local_logic_commands": local_logic,
        "candidate_commands": candidates,
        "notes": [
            "Repo currently models/labels Robinhood-facing actions; it does not contain a direct HTTP client for broker data fetches.",
            "public_mcp_read commands are route labels for trusted read surfaces, not proof of local endpoint freshness.",
            "order-style actions remain approval-gated delegate flows instead of autonomous local writes.",
        ],
    }


def summarize_android_surfaces() -> dict[str, Any]:
    return {
        "network_surfaces": [],
        "notes": [
            "Android repo is a native viewer for signal contracts, not a live market-data fetch client.",
            "Current code audit found no Retrofit/OkHttp/Ktor/browser-client data-fetch layer in app code.",
        ],
    }


def opportunities(endpoint_audit: dict[str, Any]) -> list[str]:
    probes = {probe["name"]: probe for probe in endpoint_audit.get("probes") or []}
    items = [
        "Daily bars now use the shared Yahoo chart helper; next cleanup is to extend the same low-dependency pattern to other pandas-heavy ingestion/report seams that do not truly need DataFrames.",
        "Shared retry/backoff now protects Yahoo chart probes in cockpit + audits; next step is extending the same low-drama HTTP discipline to other rate-limited public fetchers.",
        "CBOE snapshot history now persists theta, vega, rho, theo, trade prices, and bid/ask detail; next step is consuming that richer history in backtests/analytics instead of leaving it as passive storage.",
        "CBOE top-level underlying context is already flowing into cockpit source receipts; next step is surfacing more of it in operator-facing views when it improves decisions instead of adding dashboard clutter for sport.",
        "FINRA weekly persistence now carries venue concentration + publication metadata; next step is promoting HHI/top-venue share into derived dark-pool analytics rather than storing it as trivia.",
        "Factor Discord posting into one shared helper; there are currently two webhook scripts with overlapping payload logic and one had a missing-WEBHOOK bug before this audit.",
    ]
    finra = probes.get("finra_ats")
    if finra and finra.get("current_last_reported_date"):
        items.append(
            "Treat FINRA weeklySummary as a laggy convenience surface and prefer historical+local filtering for serious backfill logic; current availability looked materially stale in live probes."
        )
    return items


def build_report() -> dict[str, Any]:
    endpoint_audit = load_system_endpoint_audit()
    return {
        "generated_at": utc_now_iso(),
        "workspace_root": str(WORKSPACE_ROOT),
        "system_repo": {
            "path": str(SYSTEM_ROOT),
            "surfaces": summarize_system_surfaces(endpoint_audit),
            "freshness_findings": endpoint_audit.get("findings") or [],
        },
        "bridge_repo": {
            "path": str(BRIDGE_ROOT),
            **summarize_bridge_surfaces(),
        },
        "android_repo": {
            "path": str(ANDROID_ROOT),
            **summarize_android_surfaces(),
        },
        "opportunities": opportunities(endpoint_audit),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# SharpEdge workspace endpoint audit — {report['generated_at']}",
        "",
        "## SharpEdge-System surfaces",
    ]
    for surface in report["system_repo"]["surfaces"]:
        lines.append(f"### {surface['name']}")
        lines.append(f"- status: {surface['status']}")
        lines.append(f"- endpoint: `{surface['endpoint']}`")
        lines.append(f"- freshness_signal: `{surface['freshness_signal']}`")
        for note in surface.get("notes") or []:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## SharpEdge-System freshness findings")
    for item in report["system_repo"]["freshness_findings"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## SharpEdge-Robinhood-Bridge surfaces")
    lines.append(
        f"- route_counts: `{json.dumps(report['bridge_repo']['route_counts'], sort_keys=True)}`"
    )
    lines.append(
        f"- public_read_commands ({len(report['bridge_repo']['public_read_commands'])}): "
        + ", ".join(report["bridge_repo"]["public_read_commands"])
    )
    lines.append(
        f"- delegated_write_commands ({len(report['bridge_repo']['delegated_write_commands'])}): "
        + ", ".join(report["bridge_repo"]["delegated_write_commands"])
    )
    lines.append(
        f"- local_logic_commands ({len(report['bridge_repo']['local_logic_commands'])}): "
        + ", ".join(report["bridge_repo"]["local_logic_commands"])
    )
    lines.append(
        f"- candidate_commands ({len(report['bridge_repo']['candidate_commands'])}): "
        + ", ".join(report["bridge_repo"]["candidate_commands"])
    )
    for note in report["bridge_repo"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## SharpEdge-Android surfaces")
    lines.append("- direct network/data-fetch surfaces found: 0")
    for note in report["android_repo"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Opportunities")
    for item in report["opportunities"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
