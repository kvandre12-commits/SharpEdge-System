# SharpEdge-System Ownership

## What this repo is

SharpEdge-System is the trading-system and signal-generation repo.

It turns market data and cockpit reads into stable decision-support contracts,
especially `outputs/signal.json` and `trade_permission`.

## Owns

- Signal generation.
- Trade Gate / `trade_permission` analytics.
- Cockpit prototype generation.
- Operator artifacts and approval-state objects.
- Phone Companion contract packaging.
- Runtime proof artifacts for Golden Loop.

## Does not own

- Native Android UI rendering.
- Android activity lifecycle or Play Store packaging.
- Broker execution.
- Autonomous live orders.
- Code Puppy core runtime.
- DroidPuppy Android tool implementation.

## Stable source areas

```text
cockpit/
phone_companion/*.py
phone_companion/contracts/
docs/architecture/
tests/
README.md
OWNERSHIP.md
```

## Generated/runtime artifact areas

These change often and should not distract agents unless debugging runtime state:

```text
outputs/
phone_companion/launchers/*.json
phone_companion/observations/*.json
phone_companion/requests/*_trace.json
phone_companion/views/trading/*.json
```

## Canonical contracts

- `outputs/signal.json` uses `sharpedge.signal.v1`.
- `signal.json["trade_permission"]` uses `sharpedge.trade_permission.v1`.
- Phone Companion request/view/observation schemas live in `phone_companion/contracts/`.

## Tests

Focused local checks:

```bash
python -m pytest tests/test_trade_permission.py tests/test_phone_companion_golden_loop.py tests/test_android_viewer_export.py -q
python phone_companion/export_signal_to_android_viewer.py
```

Full test suite may require optional analytics dependencies such as pandas/numpy.

## Agent entrypoints

Read first:

1. `OWNERSHIP.md`
2. `docs/architecture/OWNERSHIP_MAP.md`
3. `docs/architecture/SYSTEM_OVERVIEW.md`
4. `docs/architecture/CURRENT_STATE.md`
5. `docs/architecture/REPO_INVENTORY.md`
6. `docs/architecture/CONTRACTS.md`

Then touch only the layer that owns the requested change.
