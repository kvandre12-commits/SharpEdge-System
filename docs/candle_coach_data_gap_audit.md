# Candle Coach Data Gap Audit

Purpose: make Candle Coach more useful as an education layer without hallucinating tape, order flow, or execution quality that SharpEdge does not actually observe.

Status date: 2026-07-28

## Executive summary

Candle Coach is doing the right safe thing today: it labels candle events, maps them to execution vectors, and explicitly warns that candles are not trade permission. The biggest senior-engineer issue is not pattern coverage. The biggest issue is data provenance and missing microstructure.

Current live coach can honestly teach:

- OHLCV candle anatomy from Yahoo 1m bars.
- Reference-level context from SharpEdge levels: ORH/ORL, PDH/PDL/PDC, VWAP, walls/pin.
- Acceptance state from recent closes around references.
- Relative-volume participation proxies.
- Dealer/gamma/OI proxy from delayed options data.
- Graph canon agreement/conflict/deference.
- Existing candle-conditioned expectancy tables in `data/spy_truth.db`.

Current live coach cannot honestly claim:

- Aggressor side from prints.
- Bid/ask imbalance.
- NBBO spread at the candle/reference level.
- Depth/queue liquidity.
- Quote replenishment/cancellation.
- Sweep detection.
- True absorption versus a wick proxy.
- Fillability/slippage for options entries.
- Real-time options tape.

## Evidence from current system

Current live `outputs/signal.json` shows Candle Coach has:

- `output_state`: `Watch`
- `data_integrity`: `partial`
- data-integrity warning: `OHLCV-only bar... missing trade-count, quote-update, spread, and feed-continuity evidence blocks trade permission.`
- missing microstructure list:
  - `aggressor_side`
  - `order_flow_imbalance`
  - `bid_ask_spread_at_level`
  - `depth_ladder`
  - `replenishment_and_cancellation`
  - `print_sequence`
  - `sweep_or_absorption_proof`

Current live example showed:

- Candle stack: shooting-star/supply-tail + inside bar + three-inside-down + ascending triangle.
- Graph canon: neutral / rotation-balance.
- Acceptance vector: CALLS, accepted above ORH.
- Volume vector: mixed/thin.
- Dealer state: positive-gamma gravity near pin 740.

That is exactly the kind of case where the coach should teach conflict: bearish local candle names do not override graph neutrality, accepted-above-ORH evidence, weak participation, or gamma pinning.

## Important finding: expectancy data exists but is not attached live

`data/spy_truth.db` contains:

- `candle_expectancy_events`: 7,674 rows
- `candle_conditioned_expectancy_matrix`: 7,469 rows
- `candle_confidence_matrix`: 7,469 rows

The live framework still emits `missing_empirical_ev` because no live lookup adapter attaches the matching `candle_confidence_matrix` row to the current candle context. This is an integration gap, not a research-data absence.

The matrix is already senior-engineer shaped: it has tiers, sample quality, realized R, target/stop rates, confidence score, deployment tier, deployment readiness, and confidence notes.

## Data classification

### A. Available real data

These are directly available now and safe to teach:

- Bar OHLCV.
- Candle body/range/wick ratios.
- Completed-bar timestamp/minute.
- VWAP from observed bars.
- Opening range, prior-day levels, current session high/low.
- Recent close acceptance around known levels.
- Relative volume versus recent median/session baseline.
- CBOE delayed options OI/IV/bid/ask proxy.
- Gamma/OI proxy, call/put walls, max pain/pin.
- Graph canon packet.
- Execution vector scores and reasons.
- Candle expectancy/confidence tables in SQLite.

### B. Proxy-only data

These can teach but must be labeled proxy:

- Participation: volume profile, aligned-volume share, path efficiency.
- Aggression: inferred from candle direction + aligned volume, not true prints.
- Absorption: wick/body/channel position, not depth/print-confirmed absorption.
- Options flow: delayed CBOE volume/OI/ATM spread proxy, not live OPRA tape.
- Dealer positioning: gamma/OI approximation, not dealer inventory.
- Fillability: ATM bid/ask spread proxy, not executable quote guarantee.

### C. Missing data

These should remain explicit blockers for any execution-grade claim:

- Trade prints with timestamps and trade sizes.
- Trade condition codes.
- Trade side classification versus NBBO midpoint/Lee-Ready style rules.
- NBBO quote stream: bid, ask, spread, quote time.
- Level-1 quote updates per candle.
- Level-2/depth ladder: queue size, replenishment, cancellations.
- Sweep/ISO-style detection.
- Live options OPRA quotes/trades.
- Broker execution fills and slippage.
- Feed-latency telemetry by source.
- Corporate-action/session-calendar validation for all symbols.

## What senior fintech help should add

### 1. Live candle expectancy adapter

Build `cockpit/candle_expectancy_adapter.py`.

Inputs:

- current Candle Coach event name/direction
- nearest reference name/relation/distance bucket
- acceptance state
- volume confirmation
- volatility/macro/dark-pool/regime/open-regime/time bucket if present

Lookup order:

1. `candle_confidence_matrix` tier 1 full match
2. tier 2 execution match
3. tier 3 core match
4. tier 4 event-only match
5. no match -> keep `missing_empirical_ev`

Output:

- matched tier
- n
- sample quality
- target-before-stop rate
- stop-before-target rate
- avg realized R
- favorable/adverse excursion
- confidence score/label
- deployment tier/readiness
- confidence notes

Safety rule: Never promote trade permission directly. It can only inform Candle Coach education and maybe add a warning/supporting note.

### 2. Data provenance and freshness packet

Every Candle Coach teaching row should expose:

- source name
- source type: real/proxy/delayed/missing
- timestamp
- age seconds/minutes
- expected max age
- fail-soft reason

Example states:

- `real_completed_bar`
- `derived_from_ohlcv`
- `delayed_options_proxy`
- `historical_expectancy_matrix`
- `missing_microstructure`

This prevents accidental hallucination when the UI gets crowded.

### 3. Microstructure readiness score, not permission score

Create a small education-only score:

- 0-25: OHLC shape only
- 25-50: OHLCV + levels + acceptance
- 50-70: plus fresh quote spread / trade count
- 70-90: plus prints/aggressor/depth/quote updates
- 90+: plus broker fill/slippage feedback

Name it something boring like `candle_evidence_readiness`, not `edge_score`, because we are not summoning a casino demon.

### 4. Optional paid/real data upgrade surfaces

If Kurtis wants execution-grade teaching later, the useful integrations are:

- Polygon/IEX/Databento/Alpaca trades and quotes for SPY underlying.
- OPRA options quotes/trades if budget allows.
- Broker fill logs for realized slippage.
- Historical intraday quote/trade replay for out-of-sample validation.

Do not add these until the adapter/provenance layer exists. More data without contracts is just expensive spaghetti.

### 5. Coach UI improvements

Add three boxes under the vector lesson:

1. `What this candle suggests`
2. `What the graph/vector stack confirms/refutes`
3. `What data is missing before this becomes execution-grade`

For current live cases, the honest message should look like:

> Candle shape suggests local rejection, but graph canon is neutral, acceptance is above ORH, volume is mixed, and positive gamma pin risk argues against treating the candle as standalone directional proof.

## Non-hallucination doctrine

Candle Coach may say:

- `suggests`
- `describes`
- `proxy indicates`
- `needs confirmation`
- `missing evidence`
- `historical matrix says, sample n=X`

Candle Coach must not say unless data exists:

- `buyers aggressively lifted offers`
- `sellers hit bids`
- `absorption confirmed`
- `large hidden buyer/seller`
- `market maker is defending`
- `sweep occurred`
- `this is executable`
- `edge confirmed`

## Highest-value next implementation

Build and test the live candle expectancy adapter first. The data already exists, and attaching it will upgrade the coach from pure education to evidence-backed education without inventing microstructure.

Second, add a provenance/missing-data panel so every fact shows whether it is real, proxy, delayed, historical, or missing.

Third, only after those contracts exist, consider real trades/quotes/depth integrations.
