CREATE TABLE IF NOT EXISTS tape_ticks (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL,
    bid REAL,
    ask REAL,
    side TEXT,
    source TEXT DEFAULT 'alpaca',
    PRIMARY KEY (ts, symbol, price, size)
);

CREATE TABLE IF NOT EXISTS orderflow_features (
    ts_bucket TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price_bucket REAL,
    buy_volume REAL DEFAULT 0,
    sell_volume REAL DEFAULT 0,
    total_volume REAL DEFAULT 0,
    delta_volume REAL DEFAULT 0,
    absorption_flag INTEGER DEFAULT 0,
    weak_push_flag INTEGER DEFAULT 0,
    liquidity_stab_flag INTEGER DEFAULT 0,
    notes TEXT,
    PRIMARY KEY (ts_bucket, symbol, price_bucket)
);
