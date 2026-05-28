# SharpEdge 2.1 — Execution Attribution & Convexity Capture Layer

## Objective

Extend SharpEdge beyond market-state prediction into execution-quality attribution.

This layer measures:

- expected move quality
- actual realized execution quality
- convexity capture efficiency
- operator decision quality
- contract structure efficiency
- state evolution during the trade

The goal is to determine not only:

- whether the market edge existed

but:

- whether the operator monetized the edge correctly.

---

# New Table

## trade_execution_log

```sql
CREATE TABLE IF NOT EXISTS trade_execution_log (

  trade_id TEXT PRIMARY KEY,

  entry_ts TEXT,
  exit_ts TEXT,

  underlying TEXT,
  underlying_price_entry REAL,
  underlying_price_exit REAL,

  contract_symbol TEXT,
  strike REAL,
  expiry TEXT,
  dte INTEGER,

  contracts INTEGER,

  entry_price REAL,
  exit_price REAL,

  realized_pnl REAL,
  realized_return_pct REAL,

  delta_entry REAL,
  gamma_entry REAL,
  theta_entry REAL,
  vega_entry REAL,
  rho_entry REAL,

  iv_entry REAL,
  chance_profit_entry REAL,

  volume_entry REAL,
  oi_entry REAL,

  open_regime_label TEXT,
  regime_id TEXT,
  pressure_state TEXT,
  transition_flag INTEGER,

  trade_reason TEXT,
  execution_notes TEXT,

  mfe REAL,
  mae REAL,

  hold_minutes REAL,

  expected_convexity REAL,
  realized_convexity REAL,
  convexity_capture_ratio REAL,

  execution_grade REAL,
  emotional_deviation_flag INTEGER,

  created_ts TEXT
);
```

---

# Core Concept

The system should eventually compare:

```text
Expected Convexity Capture
vs
Actual Convexity Capture
```

This transforms SharpEdge from:

- signal engine

into:

- execution intelligence engine.

---

# Example Interpretation

Example:

FAILED_BREAKDOWN_OPEN
+
VWAP reclaim
+
high gamma
+
0DTE near ATM

may produce:

- high realized convexity
- fast move acceleration
- short hold window

while:

far OTM continuation structures

may require:

- longer hold duration
- lower theta decay
- slower expansion profile

The system should learn these differences recursively.

---

# Future Recursive Metrics

## Operator Metrics

- hesitation delay
- premature exit
- overholding
- sizing escalation
- revenge trading
- contract selection quality
- convexity efficiency

## Market Metrics

- state persistence
- transition failure rate
- ignition continuation
- acceptance stability
- dealer support quality
- gamma acceleration quality

---

# Future Inputs

Potential ingestion sources:

- manual trade logs
- broker exports
- screenshot OCR
- Discord execution notes
- Alpaca fills
- options Greeks snapshots

---

# Strategic Importance

This layer is the bridge between:

historical expectancy

and:

real operator performance.

Without this layer:

the system only knows whether an edge existed.

With this layer:

the system learns whether the edge was successfully monetized.
