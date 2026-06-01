#!/usr/bin/env python3
import os, sqlite3
from datetime import datetime, timezone
import pandas as pd

DB_PATH=os.getenv('SPY_DB_PATH','data/spy_truth.db')
SYMBOL=os.getenv('SYMBOL','SPY')
BARS=os.getenv('INTRADAY_BARS_TABLE','spy_bars_15m')
TABLE='auction_expectancy_events'
OUT='outputs/auction_expectancy_events.csv'
REQ=['bars_daily','regime_daily','open_resolution_regime','liquidity_regime_events','overlays_daily','options_positioning_metrics']

def exists(con,t):
  return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(t,)).fetchone() is not None

def ensure(con):
  con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE}(symbol TEXT,session_date TEXT,event_type TEXT,build_ts TEXT,PRIMARY KEY(symbol,session_date,event_type))")
  con.commit()

def main():
  con=sqlite3.connect(DB_PATH)
  missing=[t for t in REQ+[BARS] if not exists(con,t)]
  if missing: raise RuntimeError(f'Missing tables: {missing}')
  ensure(con)

  feat_join=''
  exec_join=''
  trend_join=''
  feat_cols='NULL cluster_score,NULL compression_flag,NULL day_type'
  exec_cols='NULL execution_score,NULL final_bias'
  trend_cols='NULL prob_trend_fused,NULL prob_range_fused,NULL dealer_state_hint'

  if exists(con,'features_daily'):
    feat_join="LEFT JOIN features_daily fd ON fd.date=d.session_date AND fd.symbol=d.symbol"
    feat_cols='fd.cluster_score,fd.compression_flag,fd.day_type'

  if exists(con,'execution_state_daily'):
    exec_join="LEFT JOIN execution_state_daily es ON es.session_date=d.session_date AND es.symbol=d.symbol"
    exec_cols='es.execution_score,es.final_bias'

  if exists(con,'intraday_trendday_prob'):
    trend_join="LEFT JOIN intraday_trendday_prob tp ON tp.session_date=d.session_date AND tp.symbol=d.symbol AND tp.cutoff_ny=\'11:30\'"
    trend_cols='tp.prob_trend_fused,tp.prob_range_fused,tp.dealer_state_hint'

  q=f"""
  WITH daily AS (
    SELECT date session_date,symbol,open session_open,high session_high,low session_low,close session_close,
    LAG(close) OVER(PARTITION BY symbol ORDER BY date) prior_close
    FROM bars_daily WHERE symbol='{SYMBOL}'
  )
  SELECT d.symbol,d.session_date,
  COALESCE(l.regime_type,'DAILY_AUCTION') event_type,
  ((d.session_open-d.prior_close)/NULLIF(d.prior_close,0)) gap_pct,
  CASE WHEN d.session_open>d.prior_close THEN 'UP' WHEN d.session_open<d.prior_close THEN 'DOWN' ELSE 'FLAT' END gap_direction,
  d.prior_close gap_fill_level,
  d.prior_close,d.session_open,d.session_high,d.session_low,d.session_close,
  r.vol_state,r.vol_trend_state,r.dp_state,r.macro_state,r.regime_label,
  o.open_regime_label,COALESCE(o.failed_breakdown_open,0) failed_breakdown_open,
  COALESCE(o.accepted_breakdown_open,0) accepted_breakdown_open,o.regime_confidence,o.setup_dir,o.key_source,
  l.regime_type liquidity_regime_type,
  op.spot,op.gamma_wall_strike,op.pcr_oi,
  {feat_cols},
  {exec_cols},
  {trend_cols},
  '{datetime.now(timezone.utc).isoformat()}' build_ts
  FROM daily d
  LEFT JOIN regime_daily r ON r.date=d.session_date AND r.symbol=d.symbol
  LEFT JOIN open_resolution_regime o ON o.session_date=d.session_date AND o.underlying=d.symbol
  LEFT JOIN liquidity_regime_events l ON l.session_date=d.session_date AND l.underlying=d.symbol
  LEFT JOIN options_positioning_metrics op ON op.session_date=d.session_date AND op.underlying=d.symbol
  {feat_join}
  {exec_join}
  {trend_join}
  """
  df=pd.read_sql_query(q,con)
  df.to_sql(TABLE,con,if_exists='replace',index=False)
  df.to_csv(OUT,index=False)
  print(f'OK: wrote {TABLE} rows={len(df)}')

if __name__=='__main__':
  main()
