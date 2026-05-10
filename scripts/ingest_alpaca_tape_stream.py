import os
import sqlite3
import json
import asyncio
import websockets

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
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
""")

conn.commit()

async def stream():
    async with websockets.connect(WS_URL) as ws:

        auth = {
            "action": "auth",
            "key": API_KEY,
            "secret": API_SECRET
        }

        await ws.send(json.dumps(auth))

        sub = {
            "action": "subscribe",
            "trades": ["SPY"],
            "quotes": ["SPY"]
        }

        await ws.send(json.dumps(sub))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            for item in data:

                if item.get("T") == "t":

                    ts = item.get("t")
                    symbol = item.get("S")
                    price = item.get("p")
                    size = item.get("s")

                    cur.execute("""
                    INSERT INTO tape_ticks (
                        ts, symbol, price, size
                    ) VALUES (?, ?, ?, ?)
                    """, (ts, symbol, price, size))

                    conn.commit()

                    print(f"{symbol} {price} x {size}")

asyncio.run(stream())
