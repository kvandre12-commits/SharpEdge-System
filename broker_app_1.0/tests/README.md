# Broker App 1.0 Review Checklist

A surface update is acceptable only if it keeps these truths intact:

- no invented broker capabilities
- no direct live-order authority claimed for Argus
- Argus stays an MCP client / operator surface, not a scoring or strategy engine
- Argus does not calculate permission or build execution cards itself
- SharpEdge remains market-state and execution-permission authority
- Robinhood Bridge remains routing, risk, and handoff-planning authority
- live-order style actions stay behind explicit operator approval
- resource/tool names either map to real artifacts/functions or are clearly marked as draft wrappers
- thin wrapper functions stay adapters, not second-engine logic
- docs do not drift away from the real repos without evidence

## Current focused test

```bash
python -m unittest discover -s broker_app_1.0/tests -p 'test_argus_mcp_wrapper.py'
python -m unittest discover -s broker_app_1.0/tests -p 'test_argus_mcp_server.py'
```
