# SharpEdge Boring/Ruthless Contract

Status: active cleanup doctrine.

This repo has accumulated useful machinery and theatrical goblin scaffolding. The
cleanup rule is boring on purpose: do not delete source because it looks old; do
separate authoritative surfaces from disposable runtime artifacts.

## First-class live surfaces

These are allowed to be opened directly during live operation:

| Surface | Role |
|---|---|
| `cockpit/cockpit.html` | Primary SPY live read. |
| `cockpit/operator_surface.html` | Operator state/approval view. |
| `cockpit/runner_handoff_live.html` | Runner handoff view when a real handoff setup exists. |
| `cockpit/regime_nerv_split.html` | Split view: cockpit plus Regime/NERV sidecar. |
| `cockpit/regime_nerv_panel.html` | Direct Regime/NERV deck sidecar. |

Everything else is either an upstream source, a support artifact, or a demo until
it is promoted here.

## Authority boundaries

- `outputs/signal.json` is the cockpit signal packet, not an order ticket.
- Regime/NERV boards are research sidecars, not execution authority.
- Robinhood/broker actions stay approval-gated.
- Complex structures (`ratio_diagonal`, `back_ratio`, `calendar`, `diagonal`,
  income overlays, LEAPS) go through `manual_complex_structure_review`, not the
  vanilla debit-spread lane.

## Runtime artifact policy

Disposable runtime artifacts may be regenerated and may be pruned:

- `outputs/nerv/`
- `outputs/nerv_*/`
- `outputs/nerv_trade_desk/`
- `outputs/regime_cartridges/`
- `outputs/runtime_tmp/`
- cockpit loop logs/pids under `outputs/`
- stale `outputs/*.log` and `outputs/*.pid` files

Tracked research datasets and historical proof files are **not** automatically
removed by cleanup tooling. If they are too large, archive them deliberately in a
separate migration. No surprise chainsaws.

## Cleanup commands

Audit first:

```bash
python3 scripts/runtime_artifact_hygiene.py --max-age-hours 24 --largest 20
```

Apply only the allowlisted runtime prune:

```bash
python3 scripts/runtime_artifact_hygiene.py --max-age-hours 24 --apply
```

Use `--max-age-hours 0 --apply` only when intentionally clearing all allowlisted
runtime scratch artifacts. That is a broom, not a refactor.

## Boring/ruthless checklist

1. Preserve first-class surfaces.
2. Keep sidecars out of the core SPY execution loop unless they reduce decision time.
3. Prefer one canonical packet per fact.
4. Mark generated outputs disposable.
5. Add tests before adding architecture.
6. Delete only allowlisted runtime artifacts automatically.
