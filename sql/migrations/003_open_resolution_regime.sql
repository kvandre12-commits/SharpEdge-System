-- 003_open_resolution_regime.sql
CREATE TABLE IF NOT EXISTS open_resolution_regime (
  snapshot_ts TEXT NOT NULL,
  session_date TEXT NOT NULL,
  underlying TEXT NOT NULL,

  -- setup (Layer 2-ish but stored here for convenience)
  pm_open REAL,
  pm_high REAL,
  pm_low REAL,
  pm_close REAL,
  pm_return REAL,
  pm_range REAL,
  pm_range_ratio REAL,
  pm_initiative_flush INTEGER,

  -- true key level (from your daily event layer)
  break_level REAL,          -- usually prior_key_low
  flush_low REAL,            -- pm_low

  -- first RTH bar (15m: 09:30–09:45)
  rth_first_ts TEXT,
  rth_first_open REAL,
  rth_first_high REAL,
  rth_first_low REAL,
  rth_first_close REAL,

  -- resolution flags (Layer 3)
  failed_breakdown_open INTEGER,
  accepted_breakdown_open INTEGER,
  open_regime_label TEXT,
  regime_confidence REAL,
  notes TEXT,

  PRIMARY KEY (underlying, session_date)
);

CREATE INDEX IF NOT EXISTS idx_open_resolution_regime_date
  ON open_resolution_regime(underlying, session_date);
