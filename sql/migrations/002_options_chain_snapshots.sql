-- Provider-agnostic raw options snapshot table
-- Designed to match aggregate_options_positioning_metrics.py expectations

CREATE TABLE IF NOT EXISTS options_chain_snapshots (
  snapshot_ts   TEXT NOT NULL,     -- ISO8601 UTC
  session_date  TEXT NOT NULL,     -- YYYY-MM-DD (NY session date)
  underlying    TEXT NOT NULL,     -- e.g. 'SPY'
  expiry_date   TEXT NOT NULL,     -- YYYY-MM-DD
  dte           INTEGER NOT NULL,  -- calendar days to expiry
  strike        REAL NOT NULL,

  call_oi       INTEGER,
  put_oi        INTEGER,
  call_volume   INTEGER,
  put_volume    INTEGER,

  call_gamma    REAL,
  put_gamma     REAL,

  source        TEXT DEFAULT 'alpaca',

  PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike)
);

CREATE INDEX IF NOT EXISTS idx_opt_snapshots_session
  ON options_chain_snapshots (underlying, session_date, snapshot_ts);

CREATE INDEX IF NOT EXISTS idx_opt_snapshots_dte
  ON options_chain_snapshots (underlying, snapshot_ts, dte);
