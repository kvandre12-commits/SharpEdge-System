#!/usr/bin/env python3
"""Classify Robinhood capability names for SharpEdge GitHub workflows.

This is intentionally honest:
- public read/research tools are mapped from the verified public MCP source
- local active-trading actions are mapped from SharpEdge's approval-gated beta flow
- live authenticated Robinhood bridge validation is NOT performed here
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTDIR = Path("outputs")
OUT_JSON = OUTDIR / "robinhood_capability_probe.json"
OUT_TXT = OUTDIR / "robinhood_capability_probe.txt"

PUBLIC_RESEARCH = {
    "robinhood_get_accounts": "List brokerage accounts",
    "robinhood_get_portfolio": "Portfolio value and performance metrics",
    "robinhood_get_positions": "Current stock holdings",
    "robinhood_get_position": "Single stock holding by ticker",
    "robinhood_get_watchlist": "Named watchlist contents",
    "robinhood_get_quote": "Stock quote",
    "robinhood_get_fundamentals": "Fundamentals like P/E and market cap",
    "robinhood_get_historicals": "Historical price data",
    "robinhood_get_news": "Recent stock news",
    "robinhood_get_earnings": "Earnings data",
    "robinhood_get_ratings": "Analyst ratings summary",
    "robinhood_get_dividends": "Dividend payment history",
    "robinhood_get_options_positions": "Current option holdings",
    "robinhood_get_order_history": "Stock order history and status",
    "robinhood_search_symbols": "Search stocks by ticker or company name",
}

ALIASES = {
    "get_accounts": "robinhood_get_accounts",
    "get_portfolio": "robinhood_get_portfolio",
    "get_positions": "robinhood_get_positions",
    "get_position": "robinhood_get_position",
    "get_watchlist": "robinhood_get_watchlist",
    "get_quote": "robinhood_get_quote",
    "get_fundamentals": "robinhood_get_fundamentals",
    "get_historicals": "robinhood_get_historicals",
    "get_news": "robinhood_get_news",
    "get_earnings": "robinhood_get_earnings",
    "get_ratings": "robinhood_get_ratings",
    "get_dividends": "robinhood_get_dividends",
    "get_option_positions": "robinhood_get_options_positions",
    "get_options_positions": "robinhood_get_options_positions",
    "get_order_history": "robinhood_get_order_history",
    "search": "robinhood_search_symbols",
    "get_equity_positions": "robinhood_get_positions",
    "get_equity_orders": "robinhood_get_order_history",
    "get_equity_quotes": "robinhood_get_quote",
}

LOCAL_BETA_ACTIONS = {
    "create_order_draft": {
        "category": "active_trading_preview",
        "status": "local_beta_approval_gated",
        "notes": [
            "Mapped from SharpEdge robinhood_beta_execution beta_capabilities.create_order_draft.",
            "Only preview/draft intent; not autonomous live submission.",
        ],
    },
    "order_draft": {
        "category": "active_trading_preview",
        "status": "delegate_approval_gated",
        "notes": [
            "Supported as a ChatGPT Robinhood delegate task type.",
            "Handoff artifact only; no direct connector execution inside SharpEdge.",
        ],
    },
    "order_submit": {
        "category": "active_trading_write",
        "status": "delegate_operator_confirm_required",
        "notes": [
            "Supported as a ChatGPT Robinhood delegate task type.",
            "Forced to operator_confirm_required.",
        ],
    },
    "order_cancel": {
        "category": "active_trading_write",
        "status": "delegate_operator_confirm_required",
        "notes": [
            "Supported as a ChatGPT Robinhood delegate task type.",
            "Forced to operator_confirm_required.",
        ],
    },
    "order_replace": {
        "category": "active_trading_write",
        "status": "delegate_operator_confirm_required",
        "notes": [
            "Supported as a ChatGPT Robinhood delegate task type.",
            "Forced to operator_confirm_required.",
        ],
    },
    "submit_order": {
        "category": "active_trading_write",
        "status": "blocked_in_local_beta",
        "notes": [
            "Local beta explicitly blocks autonomous submit_order.",
            "Use approval-gated delegate/order preview flow instead.",
        ],
    },
    "cancel_order": {
        "category": "active_trading_write",
        "status": "blocked_without_operator_approval",
        "notes": [
            "Local beta blocks cancel without operator approval.",
        ],
    },
    "replace_order": {
        "category": "active_trading_write",
        "status": "blocked_without_operator_approval",
        "notes": [
            "Local beta blocks replace without operator approval.",
        ],
    },
}

UNVERIFIED_OR_UNSUPPORTED = {
    "get_option_orders",
    "get_option_quotes",
    "get_index_quotes",
    "get_indexes",
    "get_equity_tradability",
    "get_option_chains",
    "get_option_instruments",
    "get_option_watchlist",
    "get_watchlists",
    "create_watchlist",
    "update_watchlist",
    "get_popular_watchlists",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize(value: str) -> str:
    return str(value or "").strip().lower()


def classify(candidate: str) -> dict[str, Any]:
    normalized = normalize(candidate)
    response: dict[str, Any] = {
        "schema_version": "robinhood_capability_probe.v1",
        "created_ts": utc_now(),
        "candidate": candidate,
        "normalized": normalized,
        "live_bridge_validated": False,
        "live_bridge_note": "Authenticated tools/list validation has not been run from this workflow.",
    }

    if normalized in LOCAL_BETA_ACTIONS:
        item = LOCAL_BETA_ACTIONS[normalized]
        response.update(
            {
                "matched": True,
                "classification": item["category"],
                "status": item["status"],
                "validation_scope": "local_sharpedge_beta_or_delegate",
                "actual_tool_name": normalized,
                "summary": "Active-trading capability is approval-gated or blocked in local beta.",
                "notes": item["notes"],
            }
        )
        return response

    resolved = ALIASES.get(normalized, normalized)
    if resolved in PUBLIC_RESEARCH:
        exact = normalized == resolved
        response.update(
            {
                "matched": True,
                "classification": "research_read",
                "status": "public_source_verified" if exact else "public_source_alias_mapped",
                "validation_scope": "public_robinhood_mcp_source",
                "actual_tool_name": resolved,
                "summary": PUBLIC_RESEARCH[resolved],
                "notes": [
                    "Verified against the public read-only Robinhood MCP source, not the authenticated hosted bridge.",
                    "Use the actual tool name shown in actual_tool_name for strict inventories.",
                ],
            }
        )
        return response

    if normalized in UNVERIFIED_OR_UNSUPPORTED:
        response.update(
            {
                "matched": False,
                "classification": "unsupported_or_unverified",
                "status": "not_found_in_public_source",
                "validation_scope": "public_robinhood_mcp_source",
                "actual_tool_name": "",
                "summary": "Not found in the verified public source inventory.",
                "notes": [
                    "This does not prove the authenticated hosted Robinhood bridge lacks it.",
                    "It only means the capability was not verified in the public source or local SharpEdge beta flow.",
                ],
            }
        )
        return response

    response.update(
        {
            "matched": False,
            "classification": "unknown",
            "status": "no_local_or_public_match",
            "validation_scope": "public_source_plus_local_beta_scan",
            "actual_tool_name": "",
            "summary": "No match found in the current classifier catalog.",
            "notes": [
                "Add a mapping if this is a known alias or new capability.",
                "Authenticated Robinhood bridge validation is still the next truth source.",
            ],
        }
    )
    return response


def render_text(payload: dict[str, Any]) -> str:
    lines = [
        "ROBINHOOD CAPABILITY PROBE",
        f"Created: {payload['created_ts']}",
        f"Candidate: {payload['candidate']}",
        f"Normalized: {payload['normalized']}",
        f"Matched: {payload['matched']}",
        f"Classification: {payload['classification']}",
        f"Status: {payload['status']}",
        f"Validation scope: {payload['validation_scope']}",
        f"Actual tool name: {payload.get('actual_tool_name') or 'none'}",
        f"Live bridge validated: {payload['live_bridge_validated']}",
        "",
        f"Summary: {payload['summary']}",
        "Notes:",
    ]
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capability", help="Capability/tool/task name to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = classify(args.capability)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
