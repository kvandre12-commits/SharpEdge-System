---
name: sharpedge-data-integrity
description: SharpEdge's expensive, recurring data bugs and how to avoid re-introducing them — the CBOE-not-Alpaca open-interest source, the hollow-data (all-zero / constant) check, the symbol-clobber namespacing rule, staleness from un-wired CI builders, and the Termux python/tzdata env gotchas. Use BEFORE trusting any data column, adding an ingester, or debugging stale/empty outputs.
---

# SharpEdge Data Integrity

The bugs that ate the most time. **Validated against code + the live DB.** Read this
before you trust a column or chase a "the model is wrong" ghost — it's usually the data.

## Law #1: never trust a column is populated — CHECK it

We were burned twice by data that *looked* fine but was hollow:

- **Options open interest was ALL ZERO** (`options_chain_snapshots`, 14,734 rows,
  `call_oi/put_oi/volume` all 0). `net_gamma = Σ(γ·OI) = 0` → regime always
  "positive" → the gate's master switch was `0 >= 0`, not a market read.
- The **overlay** looked rich but was **not predictive** (different failure: real
  spread, zero edge — caught by the OOS lab, not a null check).

**Always run a hollow check on a new/critical column:**
```python
nonzero = (s != 0).mean(); std = s.std()  # plus null fraction
```
Zero nonzero or zero std = hollow. Don't build on it.

## Law #2: open interest comes from CBOE, NOT Alpaca

Alpaca's options *snapshot* endpoint **does not return open_interest** — that's why
the Alpaca ingester wrote zeros. The live cockpit already trusts **CBOE delayed
quotes** (`https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json`), which
carries real `open_interest`/`volume`/`gamma`/`iv`.

- Use `scripts/ingest_cboe_options_chain_snapshots.py` (shared store
  `scripts/options_snapshot_store.py`). `scripts/ingest_alpaca_*` is DEPRECATED
  as the OI source.
- **CBOE serves only the CURRENT snapshot** → historical OI **cannot be backfilled**.
  Real OI accrues forward from when the ingester started. Backtests of OI-derived
  gates only get teeth as days accumulate.

## Law #3: symbol-namespace every output (clobber bug)

`build_signal_strength_daily.py` once wrote `outputs/spy_signal_strength_daily.csv`
and the canonical `latest_signal_strength.csv` **regardless of `SYMBOL`** → a
`SYMBOL=WMT` run silently overwrote SPY data. Rule:
- write `outputs/{sym}_*` and `latest_signal_strength_{sym}.csv` always;
- write the **canonical un-suffixed** file ONLY when `SYMBOL == PRIMARY_SYMBOL` (SPY),
  else print a SKIP guard.

## Law #4: stale outputs usually = an un-wired builder, not a code bug

The whole SPY gate chain (`features → join_macros → signal_strength →
intraday_trendday_prob → execution_state`) was **never in `.github/workflows/ingest.yml`**
(only `ingest_wmt.yml` ran builders) → the gate fossilized. Now wired in, with the
nightly `reconcile_model_vs_reality.py`. When an output is stale: first check it's
actually *scheduled*, then check it *ran*, before suspecting logic.

## Env gotchas (Termux / this box)

- **Two pythons.** `python3` on PATH = the Code Puppy **venv WITHOUT numpy/pandas**.
  Heavy builders need **`/data/data/com.termux/files/usr/bin/python3`** (numpy/pandas/tzdata).
  Stdlib-only tools (cockpit, command deck, Bridge) run under either.
- **tzdata.** `ZoneInfo("America/New_York")` throws in bare Termux. Fix: `pip install
  tzdata` for the system python; shared helpers fall back to a fixed US-Eastern offset.
- **`disown`** is not in Termux `sh` (dash). Harmless if it scrolls by; don't rely on it.

## Quick triage when data looks wrong

1. Hollow check (nonzero fraction + std + null fraction).
2. Freshness: `max(date)` per layer — DB vs derived CSV vs what the consumer reads.
3. Source: is OI from CBOE? Is the column from the endpoint that actually serves it?
4. Scheduling: is the producing builder in `ingest.yml` and did it run?
5. Symbol: is a non-primary `SYMBOL` run clobbering canonical files?
