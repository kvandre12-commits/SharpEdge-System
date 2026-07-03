# Broker App 1.0 Docs

## Core docs

- `mcp_surface_contract.md` — Argus product boundary and surface contract
- `authority_map.md` — authority and decision boundary map
- `argus_mcp_wrapper_spec.md` — first-pass MCP wrapper endpoint contract for Argus
- `backing_matrix.md` — explicit mapping from each MCP surface to its backing component and source of truth

## Supporting review files outside this folder

Transport skeleton:
- `../mcp/server.py` — in-process MCP server skeleton
- `../mcp/tools.py` — canonical tool delegation only
- `../mcp/resources.py` — canonical resource readers only
- `../mcp/auth.py` — capability gating for exposed surface


- `../manifests/argus_mcp_manifest.json` — current Argus manifest draft
- `../bridge/real_surface_inventory.json` — real backing surface inventory
- `../tools/argus_tool_aliases.json` — tool/resource alias map
- `../schemas/README.md` — schema inventory and file naming guide
- `../prompts/chatgpt_surface_review_prompt.md` — copy/paste review prompt for ChatGPT
