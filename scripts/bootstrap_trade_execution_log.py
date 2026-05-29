import sqlite3
from pathlib import Path

DB_PATH = Path('data/spy_truth.db')

SCHEMA = '''
CREATE TABLE IF NOT EXISTS trade_execution_log (
    trade_id TEXT PRIMARY KEY,
    session_date TEXT,
    entry_ts TEXT,
    exit_ts TEXT,
    underlying TEXT,
    contract_symbol TEXT,
    strike REAL,
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
    iv_entry REAL,
    volume_entry REAL,
    oi_entry REAL,
    mfe REAL,
    mae REAL,
    hold_minutes REAL,
    market_state TEXT,
    execution_grade REAL,
    notes TEXT,
    created_ts TEXT DEFAULT CURRENT_TIMESTAMP
);
'''

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript(SCHEMA)
conn.commit()
print('trade_execution_log ready')
conn.close()
