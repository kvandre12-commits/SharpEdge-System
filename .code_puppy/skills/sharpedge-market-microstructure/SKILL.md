---
name: sharpedge-market-microstructure
description: How to read SharpEdge's sharpedge.signal.v1 (outputs/signal.json) and reason about options-dealer market microstructure — gamma regime, call/put walls, pin, max_pain, VWAP, PCR, ATM IV — and reproduce the deterministic decide() trade rule. Use whenever interpreting SPY signal cards, the cockpit, gamma data, or deciding trade/stand_down on an intraday setup.
---

# SharpEdge Market Microstructure

Operational reading guide, **validated line-by-line against the code** (not from memory).
Producers: `SharpEdge-System/cockpit/make_cockpit.py` + `cockpit/gamma.py`.
Consumer/decision: `SharpEdge-Robinhood-Bridge/.../trade_intent.py` (`decide()`).
**Interpretation only. Every live order is approval-gated (Tier C).**

## The contract: `sharpedge.signal.v1` (outputs/signal.json)

| Field | How it's actually computed (source) | Read |
|-------|-------------------------------------|------|
| `spot` | last 1m close | where we are |
| `day_chg` | `spot/day_open - 1` (%) | trend context |
| `vwap` / `vs_vwap` | cum(price*vol)/cum(vol); `(spot-vwap)/vwap*100` | >0 buyers, <0 sellers |
| `rng_pos` | `(spot-lo)/(hi-lo)*100` | 0=low, 100=high of day |
| `mom15` | `spot/close[-16] - 1` (%) over last 15 bars | short thrust |
| `vol_mult` | `avg(last 5 bar vols) / median bar vol` | conviction |
| `call_wall` | **max call-OI strike at/above spot** (OI, not gamma) | resistance |
| `put_wall` | **max put-OI strike at/below spot** (OI, not gamma) | support |
| `pcr` | total put OI / total call OI | >1 fear, <0.7 greed |
| `atm_iv` | mean(ATM call IV, ATM put IV) | expected move |
| `gamma_regime` | `positive` if `net_gamma>=0` else `negative`, where `net_gamma = Σ(callγ·callOI − putγ·putOI)` | **master switch** |
| `pin` | strike with max `magnet = callγ·callOI + putγ·putOI` (gravity center) | settle magnet |
| `max_pain` | strike minimizing total ITM option dollars | slow magnet (OPEX) |
| `setup_tag` / `setup_bias` | from `gamma_card()`, exact strings below | pre-chewed label |

> **Walls ≠ pin.** Walls are OI extremes bracketing spot. Pin is the gamma-weighted
> magnet. Different math, different roles.

## gamma_card() exact outputs (cockpit/gamma.py)

- **positive** regime → `setup_tag = "STICKY DAY (calm/chop)"`,
  `setup_bias = "FADE the edges - bet on snap-back to the magnet"`.
  Pull: spot>pin ⇒ "pull DOWN"; spot<pin ⇒ "pull UP"; else "stuck on magnet".
- **negative** regime → `setup_tag = "RUNNER DAY (wheee)"`,
  `setup_bias = "RIDE momentum - go directional, breakouts run"`.

## Two different volume thresholds (don't conflate them)

- **Cockpit display** (`synthesize()`): `vol_mult >= 1.5` = "SURGE/confirmed",
  `<= 0.7` = "THIN/fade risk", else "normal".
- **Trade decision** (`decide()`): `MIN_VOL_MULT = 1.2`. Below 1.2 = stand down.

## The authoritative decision rule — `decide(signal)`

Output is `{action: 'trade'|'stand_down', reason, intent}`. The only trade it ever
emits is **buy 1 SPY equity share, limit at spot** (no options, no shorts in v1).
Constants: `WALL_PROXIMITY_PCT=0.20`, `MIN_VOL_MULT=1.2`, `MIN_VS_VWAP=0.05`,
`MIN_MOM=0.05`. Gates evaluate **in this order** — first match wins:

1. `spot <= 0` → **stand_down** ("no spot in signal").
2. `test=True` → trade: 1-share limit buy at `spot-2` (path validation only).
3. `gamma_regime == "positive"` → **stand_down** ("sticky chop, no directional
   edge"). *Unconditional — does NOT depend on pin distance.*
4. `vol_mult < 1.2` → **stand_down** ("move not confirmed").
5. within `0.20%` of **call_wall OR put_wall** → **stand_down** ("pinned to wall").
6. `gamma_regime == "negative"` AND `vs_vwap > 0.05` AND `mom15 > 0.05`
   → **trade**: buy 1 SPY share, limit at spot ("confirmed bullish runner").
7. else → **stand_down** ("no qualifying setup").

Always name the gate that fired. Stand_down is the common, correct answer.
Conservative on purpose: "Honest > busy."

## Risk + governance (trade_intent.py)

- Allow-list **SPY only**; `MAX_EQUITY_QTY=1`; `MAX_OPTION_CONTRACTS=1`;
  `MAX_NOTIONAL_USD=1500`; kill-switch file `~/.sharpedge_kill` halts everything.
- `prepare_trade()` runs risk_check → `plan_command` (router, Tier C) → status
  `awaiting_operator_confirm`. **No silent submit.** ChatGPT Robinhood connector
  executes only AFTER the operator taps confirm.

## Worked example (validate yourself)

Live signal: gamma_regime=positive, spot 754.83, vol_mult 9.56, pin 755.
→ Gate 3 fires first: **stand_down, "positive gamma / sticky chop."**
(Volume passes, but regime gate is earlier — so the *reason* is regime, not volume.)

## How this plugs in
- Source of truth: `outputs/signal.json` (`sharpedge.signal.v1`).
- Producer: `cockpit/make_cockpit.py::write_signal`. Decision:
  `trade_intent.py::decide` + `load_signal`/`write_artifact` (→ `trade_intent_*.json`).
- Provenance + study map: Kennel + `SharpEdge-System/docs/linkedin_context_haul_2026-06-15.md`.

## Guardrails
- Interpretation only; live orders remain **operator_confirm_required**.
- If `signal.json` `ts` is stale or fields missing, say so — don't guess.
- Mirror the code; if code changes, re-validate this skill against it.
