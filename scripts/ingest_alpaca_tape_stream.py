#!/usr/bin/env python3
import os
import json
import time
import sqlite3
import asyncio
import websockets

DB_PATH = os.getenv('SPY_DB_PATH', 'data/spy_truth.db')
API_KEY = os.getenv('ALPACA_API_KEY', '').strip()
API_SECRET = os.getenv('ALPACA_API_SECRET', '').strip()

if not API_KEY or not API_SECRET:
    raise RuntimeError('Missing Alpaca credentials')

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.execute('PRAGMA journal_mode=WAL;')
conn.execute('PRAGMA synchronous=NORMAL;')

conn.execute('''
CREATE TABLE IF NOT EXISTS tape_ticks (
    ts TEXT,
    symbol TEXT,
    price REAL,
    size REAL,
    bid REAL,
    ask REAL,
    side TEXT,
    source TEXT DEFAULT 'alpaca'
)
''')

conn.commit()

WS_URL = 'wss://stream.data.alpaca.markets/v2/iex'
BATCH = []
LAST_QUOTE = {}

async def stream():
    async with websockets.connect(WS_URL) as ws:

        await ws.send(json.dumps({
            'action': 'auth',
            'key': API_KEY,
            'secret': API_SECRET
        }))

        auth_resp = await ws.recv()
        print(auth_resp)

        await ws.send(json.dumps({
            'action': 'subscribe',
            'trades': ['SPY'],
            'quotes': ['SPY']
        }))

        while True:
            raw = await ws.recv()
            data = json.loads(raw)

            for item in data:

                if item.get('T') == 'q':
                    LAST_QUOTE[item.get('S')] = {
                        'bid': item.get('bp'),
                        'ask': item.get('ap')
                    }

                if item.get('T') == 't':
                    symbol = item.get('S')
                    price = item.get('p')
                    size = item.get('s')
                    ts = item.get('t')

                    q = LAST_QUOTE.get(symbol, {})
                    bid = q.get('bid')
                    ask = q.get('ask')

                    side = None
                    if ask and price >= ask:
                        side = 'BUY'
                    elif bid and price <= bid:
                        side = 'SELL'

                    BATCH.append((
                        ts,
                        symbol,
                        price,
                        size,
                        bid,
                        ask,
                        side
                    ))

                    if len(BATCH) >= 100:
                        conn.executemany('''
                        INSERT INTO tape_ticks (
                            ts, symbol, price, size,
                            bid, ask, side
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', BATCH)

                        conn.commit()
                        print(f'Committed {len(BATCH)} ticks')
                        BATCH.clear()

                    print(f'{symbol} {price} x {size}')

            await asyncio.sleep(0.01)

asyncio.run(stream())
