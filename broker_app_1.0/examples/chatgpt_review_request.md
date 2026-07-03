# Example ChatGPT Review Request

Please review `broker_app_1.0/` as the product-facing Argus MCP client surface shell.

Your job:
- compare the manifest and docs to the real backing surfaces in `SharpEdge-System`, `SharpEdge-Robinhood-Bridge`, and `code_puppy`
- tighten naming, boundaries, and tool/resource mapping
- preserve the Discover / Read / Explain / Delegate posture
- do **not** invent broker authority or direct order execution
- prefer wrapping real artifacts/functions over duplicating logic

Start with:
- `broker_app_1.0/manifests/argus_mcp_manifest.json`
- `broker_app_1.0/docs/mcp_surface_contract.md`
- `broker_app_1.0/docs/authority_map.md`
- `broker_app_1.0/docs/argus_mcp_wrapper_spec.md`
- `broker_app_1.0/docs/backing_matrix.md`
- `broker_app_1.0/schemas/README.md`
- `broker_app_1.0/bridge/real_surface_inventory.json`
- `broker_app_1.0/tools/argus_tool_aliases.json`
- `broker_app_1.0/prompts/chatgpt_surface_review_prompt.md`
