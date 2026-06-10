# SharpEdge Agentic AI v1.0 Audit

Last audited run: `27292510732`  
Workflow: `Ingest SPY data`  
Result: `success`  
Remote output commit: `468ac69 Daily pipeline + SharpEdge 2.1 risk allocation engine`  
Audit timestamp source: repo outputs generated around `2026-06-10T17:08:00Z`

## Current Final Decision

```text
Controller decision: hold
Controller confidence: 1.0
Robinhood FVG monitor: no_trade
Risk layer state: PROBE
Pipeline warnings: none
Orders: blocked unless manually confirmed
```

The stack behaved correctly: it found a gap-fill context, recognized the gap was already filled and sample support was weak, then refused to escalate into a trade.

## Layered Intelligence Map

| Layer | Workflow step | Primary artifacts | What it adds | Current authority |
|---|---|---|---|---|
| 0 | Checkout / setup / health init | `outputs/health/run_state.txt` | Reproducible run envelope and warning ledger | Operational only |
| 1 | Ingest SPY daily bars | `data/spy_truth.db`, `outputs/spy_truth_daily.csv` | Daily OHLCV truth source | Data foundation |
| 2 | Ingest SPY intraday 15m | `spy_bars_15m` table | Intraday path and fill timing evidence | Evidence source; fail-open |
| 3 | Options positioning 0-1 / 2-3 DTE | `options_positioning_metrics`, monitor options context | Dealer/strike/OI context around current setup | Context only |
| 4 | Liquidity regime | `outputs/liquidity_regime_events.csv` | Prior keys, ATR, range regime, auction context | Feature layer |
| 5 | Open resolution regime | `open_resolution_regime` | Intraday open behavior classification | Feature layer; fail-open |
| 6 | FINRA darkpool overlay | `outputs/spy_finra_ats_weekly.csv` | Weekly off-exchange pressure proxy | Context layer; fail-open |
| 7 | FRED macro overlays | `outputs/spy_macro_overlays_daily.csv` | Macro stress/compression context | Risk context; fail-open |
| 8 | Daily regime build | `outputs/spy_regime_daily.csv` | Combines vol, macro, darkpool, compression into state labels | Regime spine |
| 9 | Auction expectancy dataset | `outputs/auction_expectancy_events.csv` | Canonical event history for gaps/breakouts/fills | Historical memory |
| 10 | Fill path classifier | `outputs/auction_fill_path_labels.csv` | Labels direct fill vs squeeze/rotation/failed behavior | Pattern classifier |
| 11 | Gap excursions | `outputs/gap_excursion_metrics.csv` | MAE/MFE/time-to-fill path risk | Edge/risk measurement |
| 12 | Conditional expectancy matrix | `outputs/conditional_expectancy_matrix.csv` | Groups historical outcomes by event/regime/path | Edge lookup table |
| 13 | Confidence matrix | `outputs/confidence_matrix.csv`, `outputs/top_confidence_states.csv` | Sample weighting, quality tiers, deployment notes | Evidence confidence |
| 14 | Risk decision layer | `outputs/risk_decision_layer.csv`, `outputs/top_risk_allocations.csv` | Converts edge/confidence/risk into deployment state and sizing caps | Risk gate |
| 15 | Regime × pressure × DTE expectancy | `outputs/edge_regime_pressure_dte.csv` | Alternative expectancy lens by regime/pressure/DTE bucket | Research lens |
| 16 | Published report | `outputs/sharpedge_2_report.md` | Human-readable market card and playbook summary | Operator explanation |
| 17 | Robinhood FVG monitor | `outputs/robinhood_fvg_monitor.json/txt` | Converts current FVG context into dry-run MCP handoff | Monitoring gate, no orders |
| 18 | Deterministic controller | `outputs/agent_controller_decision.json/txt` | Aggregates artifacts, health, risk, and monitor decision into final hold/post | Publication/action gate |
| 19 | Commit outputs | Git commit | Versioned memory of data, decisions, and artifacts | Audit trail |

## Output Inventory From Last Clean Run

| Artifact | Rows | Columns | Role |
|---|---:|---:|---|
| `spy_truth_daily.csv` | 592 | 9 | Daily OHLCV truth |
| `liquidity_regime_events.csv` | 577 | 21 | Liquidity/auction context |
| `spy_finra_ats_weekly.csv` | 125 | 11 | Darkpool overlay |
| `spy_macro_overlays_daily.csv` | 26,760 | 7 | Macro overlay rows |
| `spy_regime_daily.csv` | 577 | 21 | Daily regime spine |
| `auction_expectancy_events.csv` | 644 | 30 | Canonical historical event set |
| `auction_fill_path_labels.csv` | 644 | 12 | Fill-path labels |
| `gap_excursion_metrics.csv` | 644 | 51 | MAE/MFE/time-to-fill metrics |
| `conditional_expectancy_matrix.csv` | 233 | 27 | Edge matrix |
| `confidence_matrix.csv` | 233 | 43 | Confidence-weighted edge matrix |
| `risk_decision_layer.csv` | 9,703 | 27 | Deployment/risk matrix |
| `top_risk_allocations.csv` | 25 | 27 | Top risk candidates |
| `edge_regime_pressure_dte.csv` | 76 | 11 | Regime-pressure-DTE research lens |

## What Each Intelligence Layer Adds

### Data truth layer

The system begins with daily and intraday market facts. This layer has no opinion. Its job is to keep the rest of the stack from hallucinating off stale or missing price data.

**Adds:** OHLCV, intraday path, fill timing.  
**Failure mode:** stale data or API outage.  
**Current protection:** health warnings and fail-open for some non-core feeds.

### Context layer

Options positioning, liquidity, open behavior, darkpool, and macro overlays add market state. This is where raw price becomes "price under conditions."

**Adds:** regime labels, dealer hints, macro stress, pressure context.  
**Failure mode:** partial external feed degradation.  
**Current protection:** optional layers append warnings instead of hard failing.

### Historical memory layer

Auction events, fill-path labels, and excursion metrics transform raw history into structured examples.

**Adds:** event type, fill path, MAE/MFE, time-to-fill.  
**Failure mode:** classifications may be too broad or low sample.  
**Current protection:** later confidence/risk layers penalize low support.

### Edge layer

The conditional expectancy matrix groups outcomes by current-like states.

**Adds:** fill rate, direct fill rate, failed fill rate, expectancy, drawdown-style metrics.  
**Failure mode:** overfitting tiny buckets.  
**Current protection:** sample quality and confidence weights.

### Confidence layer

The confidence matrix tells the system whether an edge is mature enough to trust.

**Adds:** sample bucket, confidence label, deployment tier, confidence notes.  
**Failure mode:** scores are currently mixed-scale in some downstream artifacts.  
**Current protection:** deployment readiness remains conservative; no row was marked fully deployment-ready in the latest run.

### Risk decision layer

This is the capital-protection layer. It converts edge + confidence + path risk into deployment state and sizing.

**Adds:** `NO_TRADE` / `WATCH` / `PROBE` / `STANDARD` / `AGGRESSIVE`, size multiplier, risk percent, no-trade score.  
**Last fix:** date stamping was repaired so rows persist cleanly into SQLite.  
**Current result:** `PROBE`, `capital_risk_pct=0.05`, but denied reason includes `LOW_SAMPLE`.

### Monitor layer

The Robinhood FVG monitor does not trade. It packages current gap context, edge support, options context, and risk context into a dry-run handoff.

**Adds:** broker-facing watch intent and explicit blocked actions.  
**Current result:** `no_trade` because `gap_already_filled` and `low_sample_n<5`.  
**Safety:** order actions are blocked unless a human operator manually confirms.

### Controller layer

The deterministic controller is the current final agent brain. It reads the monitor, latest signal, DB risk context, and health warnings.

**Adds:** final hold/post decision and risk flags.  
**Current result:** `hold`, confidence `1.0`, risk flag `monitor_no_trade`.  
**Important:** confidence here means artifact completeness and internal consistency, not trade expectancy.

## Current Agentic AI v1.0 Verdict

This is already a layered agentic decision system, but it is not yet a fully autonomous trading AI. Good. Fully autonomous trading AI before audit discipline is how people donate money to the market with extra steps.

### What qualifies as v1.0-ready

- End-to-end scheduled/dispatchable pipeline exists.
- Data, features, edge, confidence, risk, monitor, and controller layers are separated.
- Artifacts are versioned in Git.
- Health warnings are surfaced.
- Robinhood path is dry-run and explicitly blocks orders.
- Final controller can hold even when lower layers produce attractive but unsupported edge rows.

### What must be tightened before any live-trade authority

1. **Score scale normalization**
   - Some edge artifacts expose huge `tradability_score` values while risk layer normalizes to `0..1`.
   - v1.0 should define score contracts per artifact.

2. **Duplicate top risk rows**
   - `top_risk_allocations.csv` currently repeats identical rows.
   - v1.0 should dedupe by state keys, not only by final date/symbol insert.

3. **Freshness alignment**
   - Report shows market state date `2026-05-20`, monitor gap date `2026-06-10`, and options context date `2026-05-21`.
   - v1.0 needs a freshness gate so stale context cannot masquerade as current context.

4. **Confidence semantics**
   - Controller confidence currently means "pipeline evidence completeness," while risk confidence means deployment confidence.
   - v1.0 should rename or split these to avoid operator confusion.

5. **Hard fail rules**
   - Some important steps are `continue-on-error`.
   - v1.0 should distinguish optional research overlays from mandatory safety gates.

6. **Human-readable decision contract**
   - Every final decision should state: `trade_allowed`, `broker_allowed`, `max_risk`, `reason_blocked`, `stale_inputs`.

## Proposed v1.0 Decision Contract

Every final agent output should include:

```json
{
  "symbol": "SPY",
  "decision": "hold|monitor|paper_trade|operator_confirm_required",
  "trade_allowed": false,
  "broker_order_allowed": false,
  "confidence_evidence_quality": 1.0,
  "confidence_trade_edge": 0.0,
  "risk_state": "PROBE",
  "max_capital_risk_pct": 0.0,
  "blocking_reasons": ["gap_already_filled", "low_sample", "stale_options_context"],
  "required_human_action": "none|confirm_order|review_stale_data",
  "artifact_versions": {
    "run_id": "27292510732",
    "commit": "468ac69"
  }
}
```

## Next Engineering Moves

1. Normalize all score columns and document expected ranges.
2. Add freshness gates for market state, options context, macro, and report inputs.
3. Dedupe top risk allocation outputs by state signature.
4. Split controller confidence into evidence confidence vs trade confidence.
5. Add a final `agent_v1_decision.json` contract with explicit broker permissions.
6. Keep Robinhood order placement blocked until paper-trade validation passes.

## Bottom Line

The last run proved the system can ingest, reason, risk-gate, and refuse weak setups. That is exactly the behavior we want from Agentic AI v1.0: not "always trade," but "protect the operator while compounding evidence."
