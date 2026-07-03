# Broker App 1.0

This directory is the home for the Broker App 1.0 / Argus MCP client surface draft.

Initial purpose:
- keep the operator-facing MCP client structure in Git
- anchor Argus MCP surface docs and manifests in one place
- let ChatGPT review and update the surface from the real SharpEdge + Robinhood Bridge contracts
- avoid inventing broker authority that does not exist

## First-iteration Argus job

Argus is not the trading brain.
Argus is the operator surface over existing systems.

Its first responsibilities are:
- Discover
- Read
- Explain
- Delegate

## Source-of-truth split

Executable truth does **not** live entirely in this folder.

- `SharpEdge-System` owns market state, scoring, permission, and execution-card truth
- `SharpEdge-Robinhood-Bridge` owns command routing, risk gates, and broker handoff planning
- `code_puppy` owns ChatGPT connector delegation packaging
- `broker_app_1.0/` owns the product-facing MCP client contract shell, review prompts, and mapping docs

## Directory map

- `docs/` — Argus surface contract docs
- `manifests/` — manifest-level product contract
- `bridge/` — inventory of the real backing bridge/runtime surfaces
- `tools/` — Argus tool-name to real-surface alias mapping
- `runtime/` — thin Python wrapper functions over real SharpEdge and Bridge surfaces
- `mcp/` — capability-gated MCP server skeleton that delegates to the wrappers and writes append-only request traces
- `schemas/` — machine-checkable JSON Schemas for the wrapper contract
- `prompts/` — ChatGPT review/update prompts
- `examples/` — example review requests
- `tests/` — acceptance checklist for surface updates

Key docs:
- `docs/authority_map.md` — who is allowed to decide what
- `docs/argus_mcp_wrapper_spec.md` — canonical wrapper endpoint contract
- `docs/backing_matrix.md` — explicit backing/source-of-truth matrix
- `runtime/argus_mcp_wrapper.py` — implemented thin wrapper module
- `mcp/server.py` — transport-layer MCP skeleton over the wrapper
- `mcp/tracing.py` — append-only request trace helper for observability
- `schemas/README.md` — schema inventory and naming rules
- `bridge/real_surface_inventory.json` — canonical inventory of the real backing endpoints and artifacts

## Ground rules

1. Do not invent broker capabilities.
2. Do not claim direct broker execution from Argus.
3. Preserve `operator_confirm_required` for live-order style actions.
4. Treat named Argus MCP tools/resources as a product shell until a real MCP wrapper exists.
5. Every surface update should point back to actual files or artifacts in the real backing repos.

## Review workflow

1. Read `manifests/argus_mcp_manifest.json`.
2. Read `docs/mcp_surface_contract.md`.
3. Read `docs/argus_mcp_wrapper_spec.md`.
4. Read `docs/backing_matrix.md`.
5. Read `bridge/real_surface_inventory.json`.
6. Read `tools/argus_tool_aliases.json`.
7. Read `schemas/README.md` and the per-tool schemas.
8. Use `prompts/chatgpt_surface_review_prompt.md` to ask ChatGPT for proposed updates.
9. Accept only changes that stay aligned with the real SharpEdge + Robinhood Bridge contract.
