# SharpEdge Phone Companion Edge-Token Deep Dive

## Purpose

This note captures the product/logic direction for the SharpEdge Phone Companion
before native app work resumes.

The goal is not to make another generic trading app. The goal is to preserve what
is already working in the live cockpit and turn it into a phone-native companion:

- clear market read
- conservative permission scoring
- edge-token lifecycle
- vertical-spread execution framing
- stand-down-first discipline
- reusable Yahoo/CBOE data contracts for SPY, WMT, and future watchlist symbols

SharpEdge should feel like a professional trading co-pilot that mostly tells the
operator **why not to trade** until a setup is genuinely screaming.

---

## Product doctrine

### 1. The phone companion is not the brain

The companion displays and routes the existing SharpEdge brain. It must not
recompute domain truth or invent permission.

Canonical ownership stays:

| Layer | Owns |
|---|---|
| `cockpit/` | market read, setup detection, trade permission, edge-token state |
| `outputs/signal.json` | `sharpedge.signal.v1` source contract |
| `outputs/approval_decision.json` | actual authority/permission state |
| `phone_companion/` | mobile contracts, exports, launch/import flows |
| `SharpEdge-Android` | native presentation/import/cache |
| Robinhood bridge | approval-gated broker handoff only |

The Android app can make the read beautiful. It cannot become the execution brain.

### 2. Stand-down is the default feature, not a failure

Most intraday states after big moves are balance, chop, digestion, or late-session
extension. The app must celebrate correct non-trades.

A good phone screen answers:

1. Why am I not trading?
2. What would need to change?
3. If a trade appears, why is this trade screaming?
4. Is the instrument shape a vertical spread rather than a lotto?

### 3. Professionalism beats dopamine

The companion should avoid casino language even when the signal is exciting.

Preferred language:

- `stand_down`
- `watch_only`
- `candidate`
- `confirmed`
- `invalidated`
- `expired`
- `close_first`
- `vertical_candidate`
- `risk_defined`

Avoid turning every candle into a trade. Bad puppy.

---

## What is already working in the cockpit

The favorite cockpit page works because it combines four views into one glance:

1. **Live market context**
   - spot
   - day change
   - VWAP relation
   - day range position
   - volume multiple
   - options walls
   - gamma/magnet/max pain context

2. **Permission score trend**
   - current permission score
   - direction of change
   - recent points
   - largest vertical changes since last update

3. **Execution burden verticals**
   - structure
   - acceptance
   - rejection
   - trend
   - volume
   - location
   - pressure
   - time of day
   - volatility
   - candle
   - opening auction
   - exhaustion
   - trap
   - dealer gamma
   - regime
   - compression
   - balance context

4. **Target/reachability context**
   - magnet target
   - expected move remaining
   - channel walls
   - weekly/monthly rails

This combination is the product. Do not shrink it into a single buy/sell light.

---

## Edge-token formalization

### Definition

An edge token is a tracked, directional setup claim produced by pattern/context
logic and governed by lifecycle state.

It is not just a signal. It is a small state machine:

```text
none -> candidate -> confirmed -> active -> invalidated/expired/cleared
```

### Current implementation anchor

`cockpit/edge_token_manager.py` already defines the shadow-position policy:

```text
contracts_per_token: 1
entry_on_status: confirmed
watch_only_status: candidate
exit_on_status: invalidated, expired
replacement_policy: close_first_no_same_tick_flip
max_concurrent_tokens: 1
```

That policy is correct for v1 and should be preserved.

### Token creation source

Tokens should be minted from **candlestick/setup events**, not from naked score
thresholds alone.

Good token sources:

- failed breakout / failed breakdown
- wick rejection at known level
- exhaustion after vertical run
- opening range reclaim/failure
- compression expansion with acceptance
- trend-day pullback continuation
- magnet-fade confirmation near edge

Bad token sources:

- price merely high or low
- one big candle with no level context
- raw permission score above threshold without a pattern event
- positive gamma sticky-day label alone

### Token contract shape

A token should carry enough evidence to answer "why this trade?":

```json
{
  "schema": "sharpedge.edge_token.v1",
  "token_id": "...",
  "symbol": "SPY",
  "side": "CALLS|PUTS",
  "strategy_family": "vertical_spread|watch_only",
  "source_pattern": "failed_breakout|failed_breakdown|wick_rejection|exhaustion|compression_break",
  "status": "candidate|confirmed|invalidated|expired",
  "level_name": "ORH|ORL|PDH|PDL|VWAP|magnet|channel_hi|channel_lo",
  "level_price": 747.35,
  "created_ts": "...",
  "last_seen_ts": "...",
  "confidence": 0.0,
  "permission_snapshot": {
    "score": 64,
    "label": "CAUTION",
    "bias": "NEUTRAL"
  },
  "evidence": [
    "3 closes accepted above VWAP",
    "strong bear candle near channel high",
    "balance models disagree"
  ],
  "invalidation": [
    "acceptance fails back below level",
    "token expires after N bars",
    "opposite token confirms"
  ]
}
```

The token is the bridge between pattern detection and execution discipline.

---

## Permission score trend as token telemetry

The permission trend panel should remain central. It is the pulse of the core
brain.

For phone companion v1, permission trend should show:

- current score and label
- direction: strengthening / weakening / flat
- delta over last update
- recent score points
- largest vertical changes
- active/pending token linkage
- whether the score is rising because of real acceptance/trend or noisy overlap

Important distinction:

- permission score can strengthen without creating a token
- a token can remain candidate until pattern confirmation appears
- confirmed token can still be vetoed by approval/risk/authority layers

No single number gets to cosplay as permission. Rude, but necessary.

---

## Vertical-spread execution framing

The phone companion should frame actionable options ideas as **risk-defined
vertical candidates**, not naked lotto calls/puts.

V1 does not need to auto-select strikes, but the display should prepare the
operator for vertical-spread thinking:

| Field | Meaning |
|---|---|
| `strategy_family` | `vertical_spread` by default for actionable options ideas |
| `side` | bullish call vertical or bearish put vertical |
| `expiry` | nearest practical expiry from CBOE chain/context |
| `anchor_level` | level causing the trade thesis |
| `target_level` | magnet, channel, wall, or range rail |
| `invalid_level` | level/token condition that kills the thesis |
| `premium_context` | cheap/rich vs realized move |
| `reason_to_wait` | why the system still says stand down |

The product language should be:

> "This is not a contract to buy. This is a risk-defined candidate that becomes
> interesting only if the edge token confirms and approval remains valid."

That is professional. That is also how we avoid feeding the slot machine goblin.

---

## Regime labels that need refinement

The 3-brain logic lab exposed a key design seam:

- gamma can say `STICKY DAY`
- permission verticals can say trend/acceptance is strengthening
- location can be at the upper rail
- magnet can be strategic but not reachable today
- balance models can disagree

Plain `sticky_day` is too broad for phone-native decision support.

Candidate sub-regime labels:

| Label | Meaning |
|---|---|
| `sticky_upper_rail_drift` | positive gamma/sticky context, but price is accepted high and drifting along the upper rail |
| `magnet_fade_denied_by_acceptance` | magnet pull exists, but acceptance/trend says not yet |
| `trend_day_late_extension` | late-session trend/acceptance dominates while location is stretched |
| `balance_model_disagreement` | opening/value/recent balance disagree; reduce conviction |
| `upper_edge_exhaustion_watch` | location/candle warn fade risk, but rejection/trap are not confirmed |
| `confirmed_magnet_fade` | rejection/trap confirms the fade, not just the magnet existing |
| `post_vertical_balance` | digestion after big move; default stand_down unless pattern confirms |

These labels should annotate the read. They should not override approval authority.

---

## Multi-symbol versatility: SPY and WMT

The endpoints already support the right direction:

- Yahoo 1m/daily bars for price action
- CBOE delayed options chain for options context
- output artifacts for SPY and WMT signal-strength pipelines

Phone Companion v1 should treat `symbol` as a first-class input, but keep defaults
conservative:

| Symbol class | V1 posture |
|---|---|
| SPY | primary live-read / options-context instrument |
| WMT | watchlist/proof instrument; use same artifact pattern where data quality allows |
| future symbols | require explicit watchlist registration and data-quality checks |

Do not hard-code SPY into the app UI if the contract says `symbol`. Also do not
pretend all symbols have equal options liquidity. Versatility with honesty, please.

---

## Phone companion v1 screen model

### Screen 1: Live Read

Top-level answer:

- symbol / spot / day change
- permission score + trend
- action state: stand_down / watch_only / candidate / confirmed / close
- active token state
- top reason to wait
- top reason trade could become valid

### Screen 2: Why This Trade?

Only gets loud when token is candidate or confirmed.

Must show:

- source candlestick/setup pattern
- level and target
- permission verticals that strengthened
- blockers still present
- vertical-spread framing
- invalidation rules

### Screen 3: Logic Brains Compare

A native version of the 3-up lab:

- Live Read brain
- Operator/authority brain
- Regime microscope

This is where the operator can compare logic without tab wrestling.

### Screen 4: Journal / Review

After a session or token event:

- token created
- token confirmed/invalidated
- permission score path
- whether operator acted
- outcome notes

This is how SharpEdge learns without letting journal hints override authority.

---

## V1 implementation path

### Phase 1: Contract-first system work

- keep `sharpedge.signal.v1` as primary market contract
- extend/export edge-token fields cleanly
- define optional `edge_tokens` history list separate from current position
- preserve `edge_token_position` for current v1 consumers
- add regime refinement fields as annotations, not top-level schema chaos

### Phase 2: Phone companion export

- update `export_operator_packet_to_android.py` only if the current packet needs
  extra nested fields
- keep top-level `sharpedge.operator_packet.v1` stable
- put new detail under owned nested sections such as:
  - `market_read.edge_tokens`
  - `market_read.regime_refinement`
  - `execution.edge_token_position`
  - `execution.vertical_candidate`

### Phase 3: Android presentation

- render edge token lifecycle as a first-class card
- render permission trend as telemetry, not just a number
- add "Why This Trade?" screen/card
- add vertical-spread candidate card in read-only mode
- keep all broker actions approval-gated/out of app v1

### Phase 4: Validation

- fixture: no token, stand_down
- fixture: candidate token, watch_only
- fixture: confirmed token, vertical candidate
- fixture: token invalidated, close-first state
- fixture: sticky label with trend acceptance conflict
- fixture: SPY and WMT contracts render without hard-coded symbol assumptions

---

## Non-negotiable guardrails

1. `approval_decision` remains the only authority object.
2. Android does not place trades.
3. Tokens come from setup/pattern lifecycle, not raw score alone.
4. Stand-down is a valid, successful output.
5. Vertical candidates are read-only until explicit operator approval.
6. Analytics may tighten/veto, never loosen.
7. New detail goes into nested sections, not top-level packet schema mutations.
8. SPY is primary, but symbol support must not lie about liquidity/data quality.

---

## Immediate next coding candidates

1. Add a small `regime_refinement.py` module under `cockpit/` that annotates broad
   regimes without changing core permission math.
2. Add `edge_tokens` history/export helper while preserving current
   `edge_token_position` compatibility.
3. Add phone companion fixture contracts for:
   - SPY sticky-upper-rail drift
   - SPY confirmed magnet fade
   - WMT watch-only read
4. Add Android UI card for edge-token lifecycle and "Why This Trade?".

Do these in small seams. The cockpit scoring/execution logic is the gem. Do not
refactor the gem with a chainsaw.
