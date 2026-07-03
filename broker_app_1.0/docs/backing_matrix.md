# Argus Backing Matrix

This file makes the backing path for each Argus MCP surface explicit.

If a future change cannot point to one of these backing components, it probably does not belong in Argus.

| MCP Surface | Type | Backing Component | Source of Truth | Notes |
|---|---|---|---|---|
| `sharpedge.discover_surface` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + inventory/docs files | Argus wrapper contract | Thin wrapper implemented; inventory/contract wrapper, not strategy logic |
| `sharpedge.get_latest_state` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + `~/SharpEdge-System/outputs/signal.json` | SharpEdge | Thin read adapter over latest signal artifact |
| `sharpedge.get_execution_card` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + `signal.json["trade_permission"]` + `cockpit/execution_card_builder.py` | SharpEdge | Must return authoritative card, not recalculate in Argus |
| `sharpedge.explain_permission` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + SharpEdge permission card fields | SharpEdge | Presentation/orchestration only |
| `sharpedge.prepare_broker_handoff` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + `src/sharpedge_robinhood_bridge/cockpit_adapter.py` | SharpEdge + Robinhood Bridge | Thin wrapper delegates to bridge planning/writing |
| `sharpedge.validate_handoff` | tool | `broker_app_1.0/runtime/argus_mcp_wrapper.py` + bridge validation path (`trade_intent`, `router`, `payload_contracts`) | Robinhood Bridge | Separate hard gate before downstream execution |
| `sharpedge://state/latest` | resource | `~/SharpEdge-System/outputs/signal.json` | SharpEdge | Read-only latest state |
| `sharpedge://execution/card/latest` | resource | `signal.json["trade_permission"]` | SharpEdge | Read-only latest execution card |
| `sharpedge://permission/latest` | resource | `sharpedge.trade_permission.v1` payload | SharpEdge | Read-only permission packet |
| `sharpedge://positions/latest` | resource | `~/SharpEdge-System/outputs/robinhood_live_positions.json` + `position_feedback.py` | Robinhood Bridge | Read-only positions snapshot |
| `sharpedge://handoff/latest` | resource | `~/SharpEdge-System/outputs/robinhood_execution_handoff.json` | SharpEdge-Robinhood-Bridge | Read-only latest handoff artifact |

## Boundary reminder

If you ever catch Argus doing things like:

```python
if permission > 80:
    ...
```

that logic probably belongs in SharpEdge, not here.

Argus should primarily:
- orchestrate
- present
- explain
- delegate
