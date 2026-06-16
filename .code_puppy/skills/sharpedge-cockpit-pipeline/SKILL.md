---
name: sharpedge-cockpit-pipeline
description: How SharpEdge's cockpit turns free Yahoo 1m + CBOE delayed-options feeds into signal.json and cockpit.html — the data fetch, price-action analytics, reference levels, failed-break and exhaustion setup detection, and gamma profile. Use when running/debugging the cockpit, adding a setup detector, or explaining where a signal field comes from.
---

# SharpEdge Cockpit Data Pipeline

Validated against `SharpEdge-System/cockpit/` (`make_cockpit.py`, `setups.py`,
`gamma.py`). Design: **stdlib + requests, DRY, fail-soft, decision-support only.**

## Two free, no-auth data sources
- **Yahoo 1-min intraday** `query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1m&range=1d`
  → price action. Filtered to **regular session minutes 570–960 (09:30–16:00 ET)**.
  Bar tuple = `(minute_of_session, open, high, low, close, volume)`.
- **Yahoo daily 5d** (in `setups.py::prior_day`) → prior-day high/low/close (PDH/PDL/PDC).
- **CBOE delayed options** `cdn.cboe.com/api/global/delayed_quotes/options/SPY.json`
  → OI walls, PCR, ATM IV, gamma. OCC symbol parsed by `SYM_RE` into
  `book[expiry][strike][{"C"|"P"}]`.

## Pipeline order — `make_cockpit.main()`
1. `fetch_intraday()` → rows ; `fetch_options()` → (spot, book)
2. `read_price_action(rows)` → **pa**: spot, day_open, hi, lo, rng_pos, day_chg,
   vwap, vs_vwap, mom15 (last 15 bars), vol_mult (avg last 5 / median bar).
3. `read_options(spot, book)` → **op**: call_wall (max call-OI ≥ spot),
   put_wall (max put-OI ≤ spot), pcr (ΣputOI/ΣcallOI), atm_iv (mean ATM C/P IV), exp.
4. `synthesize(pa, op)` → plain-English **lines** (each cites its numbers).
5. `reference_levels(rows)` → ORH/ORL (first 30m) + PDH/PDL/PDC.
6. `detect_failed_breaks(rows, levels)` + `detect_exhaustion(rows, pa)` → **setups**.
7. `gamma_profile(book, spot)` → **gp** ; `gamma_card(gp)` → **gcard**
   (prepended so the gamma regime sits at the top of the setups list).
8. `write_signal(pa, op, gp, gcard)` → `outputs/signal.json` (`sharpedge.signal.v1`).
9. Write `cockpit_chart.svg` + `cockpit.html` (HTML meta-refreshes for live loop).

## Setup detectors (`setups.py`)
`buffer(price) = max(0.10, price*0.0003)` (~3 bps). `RECENT_BARS = 6` — a setup
only counts if it triggered within the last 6 minutes (else "old news").

- **FAILED BREAKDOWN** (bear trap): price stabbed below ORL/PDL then reclaimed
  (last close back above level) within 6 bars → **CALLS (bullish)**.
- **FAILED BREAKOUT** (bull trap): price poked above ORH/PDH then rejected back
  below within 6 bars → **PUTS (bearish)**.
- Both scored by `depth/extension + recency`, sorted desc.
- **EXHAUSTION** (only when `rng_pos ≤ 22` lows or `≥ 78` highs): needs **≥2** of
  {volume climax ≥2.5×, momentum deceleration (|last5| < 0.6×|prior5|),
  wick ≥50% of bar, stretched ≥0.4% from VWAP} → DOWNSIDE/UPSIDE EXHAUSTION
  ("watch for reversal").

## Gamma profile (`gamma.py`)
Over nearest expiry: `net_gamma = Σ(callγ·callOI − putγ·putOI)`;
`regime = positive if net_gamma≥0 else negative`. `pin` = strike with max
`callγ·callOI + putγ·putOI` (gravity magnet). `max_pain` = strike minimizing total
ITM option dollars. `gamma_card` emits the exact tag/bias strings
("STICKY DAY (calm/chop)" / "RUNNER DAY (wheee)").

## Fail-soft contract
- `prior_day()` and `gamma_profile()` return `{}` on any failure — the cockpit
  degrades gracefully instead of crashing.
- Run: `cd ~/SharpEdge-System && python3 cockpit/make_cockpit.py` (loop for live).
- Off-hours/holidays: Yahoo returns the last session; data may be stale — check
  `signal.json` `ts`.

## How to add a setup detector (the right way)
1. Write a pure function `bars(+levels/pa) -> card dict | None` in `setups.py`
   returning `{tag, bias, kind, detail, score}`.
2. Keep it conservative + recent (respect `RECENT_BARS`).
3. Wire it into `detect_*` aggregation; it auto-renders + can feed `decide()` via
   `signal.json` if you also surface it in `write_signal`.

## Pairs with
- `sharpedge-market-microstructure` (reads signal.json + `decide()`).
- `sharpedge-execution-governance` (routes any resulting order through the gate).
