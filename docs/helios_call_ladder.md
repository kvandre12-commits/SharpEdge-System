# Helios 12:30 ET Long-Call Ladder

Helios watches for a deterministic SPY/NVDA research lead each weekday from
12:00 through 12:30 `America/New_York`. "Ladder" means a wallet-bounded basket of long OTM
calls across eligible strikes/expirations, not a call spread and never a short
option.

## Buyer policy

- Symbols: SPY and NVDA.
- Bullish trigger requires spot at least 0.05% above VWAP, positive 15-minute
  momentum of at least 0.05%, and volume ratio at least 1.2x.
- Calls only; strike must be above spot.
- Expiration must be exactly 1 or 2 calendar DTE.
- Ask must be positive and at most $1.00 per share ($100 per contract).
- Bid must be positive, quote must not be crossed, and spread must be at most
  25% of midpoint.
- Minimum volume is 10 and minimum open interest is 50.
- SPY and NVDA share one global wallet; total proposed debit across both symbols
  may not exceed $250.
- NERV volume, OI, width, IV/Greeks, and refill-target alignment rank eligible
  calls. They cannot override a failed trigger or liquidity gate.

The output is `outputs/helios_call_ladder/YYYY-MM-DD/lead_1230.json`, with a
stable copy at `outputs/helios_call_ladder/latest.json`.

## Review semantics

The deterministic reviewers are:

1. data steward
2. bullish trigger researcher
3. NERV liquidity reviewer
4. skeptic

All must approve. Missing/stale evidence causes abstention. Policy violations,
a false trigger, a non-idle cycle, weekends, or publication outside 12:00–12:30
ET cause rejection. A false trigger does not end the window: the runner checks
again on later operator refreshes until a lead is approved or 12:30 passes.

Helios approval means **research lead approved for operator review**. Every
artifact is non-authoritative, non-executable, and unable to modify
`approval_decision.json`.

## Buyer-to-seller handoff

`call_cycle_state.py` enforces:

```text
IDLE
→ ENTRY_PROPOSED
→ OPEN_POSITION              # verified fill receipt required
→ SELLER_MANAGING            # begins at +20%
→ EXIT_PENDING_CONFIRM
→ CLOSED                     # verified close receipt required
→ COOLDOWN
→ IDLE                       # trigger reset required
```

## Thesis ownership

A verified fill hands the owner two distinct objects:

1. authoritative position facts: exact contracts, quantities, debit, and fill receipt;
2. the immutable, content-hashed Helios entry thesis: trigger evidence, refill
   target, source hashes, strategy, and seller policy as they existed at entry.

The thesis is context, not authority. A broken thesis may accelerate an exit,
but a valid thesis cannot override hard risk, expiry, liquidity, or profit-management
rules and cannot authorize adding or rebuying. If the thesis is missing, stale,
or unknown, the owner falls back to its independent rules rather than assuming
"hold."

Seller review bands:

- +20%: protect and review
- +35%: tighten exit review
- +55%: strong take-profit review

These are review states, not autonomous sell commands. One owned contract can
only be sold once; multiple filled ladder contracts may later be managed as
separate owned quantities.

## Runtime

The always-running operator refresh calls:

```bash
python3 -m scripts.agents.helios_call_ladder_runner
```

Outside the 12:00–12:30 ET window it exits without network work. During the
window it reuses a dedicated SPY/NVDA NERV snapshot for up to five minutes,
captures current intraday features, and keeps evaluating until it publishes an
approved wallet reservation or the window closes. An approved proposal latches
the shared wallet against duplicate or cross-symbol entries.

Manual bounded validation:

```bash
python3 -m scripts.agents.helios_call_ladder_runner --force
```

A forced off-hours run should reject or abstain; it must not create executable
authority.

## Current capability boundary

The Robinhood bridge still requires operator confirmation for order writes. It
has not been verified here to autonomously buy or sell these calls. Broker quote,
buying-power, contract, and final limit-price checks remain mandatory at the
execution surface. Free NERV/Yahoo data is research-only and delayed or of
unknown delay.
