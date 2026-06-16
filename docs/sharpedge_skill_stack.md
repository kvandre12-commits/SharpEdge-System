# SharpEdge Skill Stack — Index

Code Puppy skills that teach any SharpEdge session how the system actually works.
Each was **validated against running code**, not written from memory.
Skills live in `~/.code_puppy/skills/<name>/SKILL.md` (auto-discovered).

## Built & validated

| # | Skill | Covers | Validated by |
|---|-------|--------|--------------|
| 1 | `sharpedge-market-microstructure` | `signal.json` (sharpedge.signal.v1) fields + the deterministic `decide()` trade rule (gates, constants, 1-share-SPY-only output) | Ran `decide(load_signal())` → `stand_down` "positive gamma / sticky chop" as predicted |
| 2 | `sharpedge-execution-governance` | Command routing taxonomy (public_mcp_read / chatgpt_delegate / custom_logic_local / custom_logic_required), approval tiers, risk guardrails, no-silent-submit law | Ran `plan_command()` across 5 commands → routes/policies matched 1:1 |
| 3 | `sharpedge-cockpit-pipeline` | Yahoo 1m + CBOE feeds → price action, reference levels, failed-break & exhaustion detectors, gamma profile → `signal.json` + `cockpit.html` | Ran `make_cockpit.py` live → regenerated signal.json, output matched doc |
| 4 | `sharpedge-operator-agents` | scripts/agents: safety contract, deterministic controller, 4 canonical objects (workflow_state/execution_plan/approval_decision/journal), brief/dashboard/review/beta/hints | 12/12 unittest pass; **fixed a real test-isolation bug** in test_robinhood_beta_execution |
| 5 | `sharpedge-analytics-pipeline` | scripts/ (37) + spy_truth.db (28 tables): ingest → features → regimes → expectancy/backtests → confidence/calibration → signal_strength/monitor → cards | Spot-checked script→output/table claims (signal_strength.csv, fvg_monitor.json, 35/37 use SPY_DB_PATH) |

## How they relate
`cockpit-pipeline` (produces the signal) → `market-microstructure` (decides
whether to trade) → `execution-governance` (routes/gates any resulting order).

## Validated ground-truth constants (quick ref)
- `decide()`: WALL_PROXIMITY_PCT=0.20, MIN_VOL_MULT=1.2, MIN_VS_VWAP=0.05, MIN_MOM=0.05.
  Positive gamma = unconditional stand_down. Only trade = buy 1 SPY equity @ spot.
- Risk: SPY-only, qty≤1, contracts≤1, notional≤$1500, kill-switch `~/.sharpedge_kill`.
- Cockpit display vol thresholds (1.5/0.7) differ from decide() (1.2) — by design.

## Status: full stack built (5/5, all validated)
The live trade path (cockpit → decide → route), the operator/authority layer, and
the offline research/analytics layer are all covered and code-verified.

> Principle: only build a skill when a real session would need it. Each skill was
> validated against running code (or a passing test suite), not written from memory.
> Bonus: validating #4 surfaced & fixed a real non-hermetic test.
