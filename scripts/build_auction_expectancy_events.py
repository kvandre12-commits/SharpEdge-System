#!/usr/bin/env python3
import os, sqlite3
from datetime import datetime, timezone
import pandas as pd
import numpy as np

DB_PATH=os.getenv('SPY_DB_PATH','data/spy_truth.db')
SYMBOL=os.getenv('SYMBOL','SPY')
BARS=os.getenv('INTRADAY_BARS_TABLE','spy_bars_15m')
TABLE='auction_expectancy_events'
OUT='outputs/auction_expectancy_events.csv'
REQ=['bars_daily','regime_daily','open_resolution_regime','liquidity_regime_events','overlays_daily','options_positioning_metrics']

def exists(con,t):
  return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(t,)).fetchone() is not None

def ensure(con):
  con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE}(symbol TEXT,session_date TEXT,event_type TEXT,gap_pct REAL,gap_direction TEXT,gap_fill_level REAL,fill_completed INTEGER,fill_ts TEXT,time_to_fill_minutes REAL,prior_close REAL,session_open REAL,session_high REAL,session_low REAL,session_close REAL,vol_state TEXT,vol_trend_state TEXT,dp_state TEXT,macro_state TEXT,regime_label TEXT,open_regime_label TEXT,liquidity_regime_type TEXT,spot REAL,gamma_wall_strike REAL,pcr_oi REAL,build_ts TEXT,PRIMARY KEY(symbol,session_date,event_type))")
  con.commit()

def main():
  con=sqlite3.connect(DB_PATH)
  missing=[t for t in REQ+[BARS] if not exists(con,t)]
  if missing: raise RuntimeError(f'Missing tables: {missing}')
  ensure(con)
  q=f"""
  WITH daily AS (
    SELECT date session_date,symbol,open session_open,high session_high,low session_low,close session_close,
    LAG(close) OVER(PARTITION BY symbol ORDER BY date) prior_close
    FROM bars_daily WHERE symbol='{SYMBOL}'
  ), fills AS (
    SELECT d.session_date,MIN(i.ts) fill_ts
    FROM daily d
    LEFT JOIN {BARS} i ON i.session_date=d.session_date AND i.symbol=d.symbol
      AND i.low<=d.prior_close AND i.high>=d.prior_close
    GROUP BY d.session_date
  ), ov AS (
    SELECT date session_date,symbol,
    MAX(CASE WHEN overlay_type='darkpool' THEN overlay_strength END) darkpool,
    MAX(CASE WHEN overlay_type='vix' THEN overlay_strength END) vix
    FROM overlays_daily GROUP BY date,symbol
  )
  SELECT d.symbol,d.session_date,
  COALESCE(l.regime_type,'DAILY_AUCTION') event_type,
  ((d.session_open-d.prior_close)/NULLIF(d.prior_close,0)) gap_pct,
  CASE WHEN d.session_open>d.prior_close THEN 'UP' WHEN d.session_open<d.prior_close THEN 'DOWN' ELSE 'FLAT' END gap_direction,
  d.prior_close gap_fill_level,
  CASE WHEN f.fill_ts IS NULL THEN 0 ELSE 1 END fill_completed,
  f.fill_ts,
  CASE WHEN f.fill_ts IS NULL THEN NULL ELSE (julianday(f.fill_ts)-julianday((SELECT MIN(ts) FROM {BARS} b WHERE b.session_date=d.session_date)))*1440 END time_to_fill_minutes,
  d.prior_close,d.session_open,d.session_high,d.session_low,d.session_close,
  r.vol_state,r.vol_trend_state,r.dp_state,r.macro_state,r.regime_label,
  o.open_regime_label,l.regime_type liquidity_regime_type,
  op.spot,op.gamma_wall_strike,op.pcr_oi,
  '{datetime.now(timezone.utc).isoformat()}' build_ts
  FROM daily d
  LEFT JOIN fills f ON f.session_date=d.session_date
  LEFT JOIN regime_daily r ON r.date=d.session_date AND r.symbol=d.symbol
  LEFT JOIN open_resolution_regime o ON o.session_date=d.session_date AND o.underlying=d.symbol
  LEFT JOIN liquidity_regime_events l ON l.session_date=d.session_date AND l.underlying=d.symbol
  LEFT JOIN ov ON ov.session_date=d.session_date AND ov.symbol=d.symbol
  LEFT JOIN options_positioning_metrics op ON op.session_date=d.session_date AND op.underlying=d.symbol
  """
  df=pd.read_sql_query(q,con)
  if df.empty: raise RuntimeError('0 rows generated.')
  df.to_sql(TABLE,con,if_exists='replace',index=False)
  os.makedirs('outputs',exist_ok=True)
  df.to_csv(OUT,index=False)
  print(f'OK: wrote {TABLE} rows={len(df)}')
  print(f'OK: wrote {OUT}')
  con.close()

if __name__=='__main__':
  main()
