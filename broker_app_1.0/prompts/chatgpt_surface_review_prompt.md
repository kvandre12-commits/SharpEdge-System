# ChatGPT Surface Review Prompt

You are reviewing `broker_app_1.0/` as the product-facing Argus MCP client surface for SharpEdge.

## Goal

Update the Argus surface so it reflects the **real** SharpEdge + Robinhood Bridge contract instead of speculative app behavior.

## Non-negotiable rules

- Do not invent broker capabilities.
- Do not claim direct broker execution from Argus.
- Preserve explicit operator approval for live-order style actions.
- Treat SharpEdge as execution-permission authority.
- Treat SharpEdge-Robinhood-Bridge as routing / broker handoff authority.
- Prefer wrapping existing artifacts and functions over rebuilding logic.
- Preserve the Discover / Read / Explain / Delegate posture for first-iteration Argus.

## Review inputs

Read these first:

- `broker_app_1.0/manifests/argus_mcp_manifest.json`
- `broker_app_1.0/docs/mcp_surface_contract.md`
- `broker_app_1.0/docs/authority_map.md`
- `broker_app_1.0/docs/argus_mcp_wrapper_spec.md`
- `broker_app_1.0/docs/backing_matrix.md`
- `broker_app_1.0/schemas/README.md`
- `broker_app_1.0/bridge/real_surface_inventory.json`
- `broker_app_1.0/tools/argus_tool_aliases.json`

Then verify against the real backing repos:

### SharpEdge-System
- `OWNERSHIP.md`
- `cockpit/execution_card_builder.py`
- `outputs/signal.json` when present

### SharpEdge-Robinhood-Bridge
- `src/sharpedge_robinhood_bridge/catalog.py`
- `src/sharpedge_robinhood_bridge/trade_intent.py`
- `src/sharpedge_robinhood_bridge/cockpit_adapter.py`
- `src/sharpedge_robinhood_bridge/payload_contracts.py`
- `src/sharpedge_robinhood_bridge/position_feedback.py`

### code_puppy
- `code_puppy/plugins/chatgpt_robinhood_delegate/`

## Deliverables

Return:
1. a short summary of what is already real
2. a short summary of what is still just a product-shell wrapper
3. proposed edits to manifest/docs/tool mappings
4. any missing MCP wrapper endpoints that should be added later
5. any schema drift or contract mismatch you see
6. any wording that overclaims authority and should be corrected

## Output style

Be strict, concrete, and repo-backed.
If a capability is not implemented, say so plainly.
