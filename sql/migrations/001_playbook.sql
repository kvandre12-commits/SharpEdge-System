-- 001_playbook.sql
-- Codifies: Failed Breakdown Long (options-friendly) + DTE playbook + auto DTE fields

-- -----------------------
-- Rule registry
-- -----------------------
CREATE TABLE IF NOT EXISTS signal_rules (
  rule_id        TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  direction      TEXT NOT NULL,
  timeframe      TEXT NOT NULL,
  params_json    TEXT NOT NULL,
  notes          TEXT,
  created_ts     TEXT DEFAULT (datetime('now'))
);

INSERT OR REPLACE INTO signal_rules (
  rule_id, name, direction, timeframe, params_json, notes
) VALUES (
  'FB_LONG_OPT_V1',
  'Failed Breakdown Long (Options, acceptance stop)',
  'LONG',
  '15m',
  json_object(
    'lookback_bars_support', 4,
    'reclaim_window_bars', 3,
    'use_pullback_entry', 1,
    'pullback_window_bars', 3,
    'stop_mode', 'close_below_support',
    'stop_confirm_bars', 2,
    'use_catastrophic_stop', 1
  ),
  'Sweep below support -> reclaim. Prefer pullback entry. Invalidate on acceptance (2 consecutive closes below support). Optional catastrophic stop below sweep low.'
);

-- -----------------------
-- Options playbook (DTE tables)
-- -----------------------
CREATE TABLE IF NOT EXISTS options_playbook (
  playbook_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_id          TEXT NOT NULL,
  dte_bucket       TEXT NOT NULL,        -- '0-1', '2-3', '5-7'
  entry_timing     TEXT NOT NULL,
  strike_bias      TEXT NOT NULL,
  stop_logic       TEXT NOT NULL,
  profit_plan      TEXT NOT NULL,
  expected_behavior TEXT NOT NULL,
  sizing_guidance  TEXT NOT NULL,
  notes            TEXT
);

DELETE FROM options_playbook WHERE rule_id = 'FB_LONG_OPT_V1';

INSERT INTO options_playbook
(rule_id,dte_bucket,entry_timing,strike_bias,stop_logic,profit_plan,expected_behavior,sizing_guidance,notes)
VALUES
('FB_LONG_OPT_V1','0-1',
 'Enter only after reclaim close OR first pullback hold; no anticipation',
 'ATM or slightly ITM calls',
 'Exit on FIRST close below reclaimed support (no tolerance)',
 'Partial scalp into impulse; runner only if expansion is immediate',
 'Price should move away from support quickly; chop is failure',
 'Small size; high gamma/decay',
 'Use only when reclaim closes strong; avoid dead chop'),

('FB_LONG_OPT_V1','2-3',
 'Prefer pullback entry that tags support and closes above',
 'ATM or 1 strike ITM calls',
 'Invalidate on acceptance: 2 consecutive closes below support',
 'Multi-contract ladder: trim 1R, 2R, hold runner',
 'Price may rotate but must not accept below reclaimed structure',
 'Core size; best balance of gamma + forgiveness',
 'Highest expectancy expression'),

('FB_LONG_OPT_V1','5-7',
 'Enter after daily confirmation (close holds above reclaimed level)',
 'ITM calls or call spreads',
 'Exit only on DAILY acceptance below support',
 'Trend hold; scale out into strength',
 'Intraday noise ok; multi-day higher highs matter',
 'Moderate size; higher delta/lower gamma',
 'Best when trap aligns with regime shift');

-- -----------------------
-- Trade signals (persistent objects)
-- -----------------------
CREATE TABLE IF NOT EXISTS trade_signals (
  signal_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_id        TEXT NOT NULL,
  symbol         TEXT NOT NULL,
  sweep_ts       TEXT NOT NULL,
  entry_ts       TEXT NOT NULL,
  entry_price    REAL NOT NULL,
  prior_support  REAL NOT NULL,
  sweep_low      REAL NOT NULL,
  dte_bucket     TEXT,
  dte_reason     TEXT,
  plan_json      TEXT,
  created_ts     TEXT DEFAULT (datetime('now'))
);

-- -----------------------
-- Signal generator view (structure-based on spy_bars_15m)
-- Requires table from ingest_spy_intraday_alpaca.py
-- -----------------------
CREATE VIEW IF NOT EXISTS v_failed_breakdown_long_opt AS
WITH ordered AS (
  SELECT
    ts, session_date, symbol, open, high, low, close, volume,
    MIN(low) OVER (
      PARTITION BY symbol, session_date
      ORDER BY ts
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) AS prior_support
  FROM spy_bars_15m
  WHERE symbol = 'SPY'
),
sweeps AS (
  SELECT *, low AS sweep_low
  FROM ordered
  WHERE prior_support IS NOT NULL AND low < prior_support
),
reclaims AS (
  SELECT
    s.ts AS sweep_ts,
    s.session_date,
    s.prior_support,
    s.sweep_low,
    o.ts AS reclaim_ts,
    o.close AS reclaim_close
  FROM sweeps s
  JOIN ordered o
    ON o.symbol = s.symbol
   AND o.session_date = s.session_date
   AND o.ts > s.ts
   AND o.ts <= datetime(s.ts, '+45 minutes')
  WHERE o.close > s.prior_support
),
first_reclaim AS (
  SELECT *
  FROM (
    SELECT r.*, ROW_NUMBER() OVER (PARTITION BY sweep_ts ORDER BY reclaim_ts) AS rn
    FROM reclaims r
  )
  WHERE rn = 1
),
pullback_entry AS (
  SELECT
    fr.sweep_ts,
    fr.session_date,
    fr.prior_support,
    fr.sweep_low,
    fr.reclaim_ts,
    o.ts AS entry_ts,
    o.close AS entry_price
  FROM first_reclaim fr
  JOIN ordered o
    ON o.symbol='SPY'
   AND o.session_date = fr.session_date
   AND o.ts > fr.reclaim_ts
   AND o.ts <= datetime(fr.reclaim_ts, '+45 minutes')
  WHERE o.low <= fr.prior_support
    AND o.close > fr.prior_support
),
first_entry AS (
  SELECT *
  FROM (
    SELECT p.*, ROW_NUMBER() OVER (PARTITION BY sweep_ts ORDER BY entry_ts) AS rn2
    FROM pullback_entry p
  )
  WHERE rn2 = 1
)
SELECT
  'FB_LONG_OPT_V1' AS rule_id,
  'SPY' AS symbol,
  sweep_ts,
  entry_ts,
  entry_price,
  prior_support,
  sweep_low
FROM first_entry;

-- -----------------------
-- Trade card view (join signal -> selected DTE plan)
-- -----------------------
CREATE VIEW IF NOT EXISTS v_trade_cards AS
SELECT
  t.signal_id,
  t.rule_id,
  t.symbol,
  t.sweep_ts,
  t.entry_ts,
  t.entry_price,
  t.prior_support,
  t.sweep_low,
  t.dte_bucket,
  t.dte_reason,
  p.entry_timing,
  p.strike_bias,
  p.stop_logic,
  p.profit_plan,
  p.expected_behavior,
  p.sizing_guidance,
  p.notes
FROM trade_signals t
LEFT JOIN options_playbook p
  ON p.rule_id = t.rule_id
 AND p.dte_bucket = t.dte_bucket;
