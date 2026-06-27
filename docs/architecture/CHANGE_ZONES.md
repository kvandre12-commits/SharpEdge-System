# SharpEdge Change Zones

This file defines **where change is safe** and **where proof requirements rise**.
The goal is simple: protect the cockpit gem while keeping the repo fluid enough
for real iteration.

## Zone table

| Zone | Purpose | Change policy |
|---|---|---|
| Red: Core decision engine | Converts market state into setups, permission, conviction, bias, targeting, and receipt semantics. | **High proof required.** Behavioral changes only with focused tests, evidence, and explicit intent. No casual cleanup that changes trading meaning. |
| Orange: Contract layer | Public APIs, exported objects, JSON contracts, and cross-repo payload shapes. | Preserve compatibility unless deliberately versioned. Any rename, deletion, or shape change needs consumer audit. |
| Yellow: Presentation / consumers | Cockpit rendering, dashboards, and Android-facing packaging/exports. | Safer to iterate, but do not silently mutate contract meaning or decision semantics. |
| Green: Infrastructure / support | Docs, launchers, audit helpers, and non-decision plumbing. | Safest area for cleanup and refactor. Keep it honest, but this is the lowest-risk lane. |

## Red zone: Core decision engine

These files are part of the trading brain and should be treated as protected:

- `cockpit/make_cockpit.py`
- `cockpit/setups.py`
- `cockpit/trade_permission.py`
- `cockpit/execution_vector_engine.py`
- `cockpit/trade_permission_context.py`
- `cockpit/execution_vector_weights.py`
- `cockpit/execution_vector_primitives.py`
- `cockpit/setup_conviction.py`
- `cockpit/gate_workflows.py`
- `cockpit/targeting.py`
- `cockpit/decision_receipts.py`
- `cockpit/setup_event_lifecycle.py`

### Why targeting and receipts stay in red

They are **not** just formatting.

- `targeting.py` encodes strategic objective semantics.
- `decision_receipts.py` defines how the system records and explains decision state over time.
- `setup_event_lifecycle.py` defines how setup state changes are interpreted.

Changing those files can change what the system **means**, not just how it
looks.

### High-risk seam: acceptance / rejection / trap

Current doctrine from the live engine:

- `acceptance_score`, `rejection_score`, and `trap_score` are shown as separate rows
- score aggregation uses a shared score bucket via:

```python
max(acceptance_score, rejection_score, trap_score)
```

- bias aggregation still treats them separately

That asymmetry may be intentional. Do **not** casually "simplify" it.

## Orange zone: Contract layer

These surfaces are relied on by downstream consumers and should remain stable
unless explicitly versioned:

- `docs/architecture/CONTRACTS.md`
- `outputs/signal.json` shape
- `sharpedge.trade_permission.v1` object shape
- `sharpedge.setup_conviction.v1` object shape
- decision receipt payload shape
- permission score trend payload shape
- phone companion contract readers
- Android import/export payload expectations

### Contract rule

If a field name, nesting shape, or semantic meaning changes:

1. identify consumers first
2. update tests
3. version deliberately if compatibility is not preserved

Do not create near-duplicate fields just because a rename feels nicer.

## Yellow zone: Presentation / consumers

These layers are downstream consumers of the core decision objects:

- `cockpit/live_read_view.py`
- `cockpit/live_chart_svg.py`
- `cockpit/make_operator_surface.py`
- `cockpit/dashboard_runtime.py`
- `cockpit/weekly_context_chart.py`
- `cockpit/monthly_context_chart.py`
- `phone_companion/export_signal_to_android_viewer.py`
- `phone_companion/export_operator_packet_to_android.py`
- `phone_companion/launchers/*`

These files are safer to iterate on **as long as** they preserve the meaning of
core contracts and scores.

## Green zone: Infrastructure / support

Typical low-risk areas:

- `docs/**`
- launchers
- audit scripts
- helper utilities that do not change decision semantics
- local workflow wrappers

This is the preferred lane for cleanup when the core engine is not the target.

## Generated artifacts are not doctrine

The following are useful evidence, but they are not primary source logic:

- `outputs/**`
- `outputs/permission_receipts_spy.jsonl`
- `outputs/*.csv`
- `outputs/*.md`
- `outputs/health/*.json`
- `data/spy_truth.db`

Treat them as runtime artifacts, audit evidence, or derived state — not as the
place where architecture truth lives.

## Proof expectations by zone

### For red-zone core changes

Expect focused proof such as:

```bash
python -m pytest tests/test_trade_permission.py -q
```

Plus any directly affected tests for setups, targeting, receipts, or consumer
contracts.

### For orange-zone contract changes

Expect:

- consumer audit
- fixture/test updates
- explicit compatibility decision

### For yellow-zone presentation changes

Expect:

- no silent contract drift
- render/export tests where available

### For green-zone support changes

Expect:

- basic honesty
- minimal regression risk

## Working doctrine

When in doubt:

1. change the green zone first
2. then yellow
3. then orange with explicit consumer awareness
4. touch red only when the objective is genuinely decision-engine work and proof exists
