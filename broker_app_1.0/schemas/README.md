# Argus MCP Schemas

This directory contains the machine-checkable JSON Schemas for the first-pass Argus MCP wrapper surface.

## Canonical v0 tools

- `sharpedge.discover_surface`
- `sharpedge.get_latest_state`
- `sharpedge.get_execution_card`
- `sharpedge.explain_permission`
- `sharpedge.prepare_broker_handoff`
- `sharpedge.validate_handoff`

Each tool has two schemas:
- `<tool>.input.schema.json`
- `<tool>.output.schema.json`

Example:
- `sharpedge.get_execution_card.input.schema.json`
- `sharpedge.get_execution_card.output.schema.json`

## Shared defs

- `common.schema.json` — shared enums, response metadata, authority names, and common entry shapes

## Contract rule

These schemas enforce the wrapper contract.
They do **not** move business logic into Argus.

Argus still:
- orchestrates
- presents
- explains
- delegates

SharpEdge still decides.
Bridge still translates.
Operator still approves.
