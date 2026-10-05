#!/usr/bin/env python3
"""
Ingest Current Events (SharpEdge 2.0 - Context Layer).

Pulls the last ~24h of market-relevant headlines from the GDELT DOC 2.0 API
(free, no API key) and stores them in `news_daily` with a lightweight
keyword tone score in [-1, 1].

Fail-open by design: GDELT rate-limits (HTTP 429) and hiccups. Any failure
logs a warning and exits 0 so the scheduled pipeline never breaks on news.
Set NEWS_FAIL_OPEN=0 to make failures fatal (not recommended for cron).

Schema:
    news_daily(date, symbol, title, source, url, tone)
    PRIMARY KEY (date, symbol, url)

Usage:
    python scripts/ingest_current_events.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/ingest_current_events.py
"""

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
FAIL_OPEN = os.getenv("NEWS_FAIL_OPEN", "1").strip() == "1"
TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "30"))
MAX_RECORDS = int(os.getenv("NEWS_MAX_RECORDS", "50"))

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '("stock market" OR "S&P 500" OR "Federal Reserve" OR "Wall Street")'

POSITIVE_WORDS = {
    "rally", "rallies", "rallied", "surge", "surged", "surges", "gain", "gains",
    "gained", "record high", "record highs", "optimism", "optimistic", "rebound",
    "rebounds", "rebounded", "soar", "soared", "soars", "jump", "jumped",
    "jumps", "breakthrough", "deal", "rate cut", "cuts rates", "bullish",
    "recovery", "recovering", "eases", "cooling inflation", "beats", "beat",
    "upgrade", "upgraded", "confidence",
}
NEGATIVE_WORDS = {
    "crash", "crashed", "crashes", "plunge", "plunged", "plunges", "tumble",
    "tumbled", "tumbles", "fear", "fears", "recession", "selloff", "sell-off",
    "sell off", "slump", "slumped", "warning", "warns", "bearish", "sink",
    "sinks", "sank", "crisis", "layoffs", "downgrade", "downgraded", "miss",
    "misses", "default", "collapse", "collapsed", "panic", "turmoil",
    "invasion", "war", "sanctions",
}


def score_tone(title: str) -> float:
    """Keyword tone in [-1, 1]. 0 = no signal words found."""
    text = f" {title.lower()} "
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    denom = pos + neg
    if denom == 0:
        return 0.0
    return round((pos - neg) / denom, 3)


def parse_gdelt_date(seendate: str) -> str:
    """GDELT seendate like '20261005T013000Z' -> '2026-10-05'. Falls back to today UTC."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", seendate or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return datetime.now(timezone.utc).date().isoformat()


def fetch_articles() -> list:
    params = urllib.parse.urlencode({
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": MAX_RECORDS,
        "timespan": "24h",
        "format": "json",
    })
    req = urllib.request.Request(
        f"{GDELT_URL}?{params}",
        headers={"User-Agent": "SharpEdge-ContextLayer/2.0"},
    )
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                payload = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 1:
                time.sleep(6)  # GDELT asks for >=5s between requests
                continue
            raise
    articles = payload.get("articles") or []
    out = []
    for a in articles:
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue
        out.append({
            "date": parse_gdelt_date(a.get("seendate") or ""),
            "title": title,
            "source": (a.get("domain") or "").strip(),
            "url": url,
            "tone": score_tone(title),
        })
    return out


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS news_daily (
          date   TEXT NOT NULL,
          symbol TEXT NOT NULL,
          title  TEXT NOT NULL,
          source TEXT,
          url    TEXT NOT NULL,
          tone   REAL NOT NULL DEFAULT 0.0,
          PRIMARY KEY (date, symbol, url)
        );
        """
    )
    con.commit()


def store(con: sqlite3.Connection, symbol: str, articles: list) -> int:
    before = con.execute("SELECT COUNT(*) FROM news_daily").fetchone()[0]
    con.executemany(
        """
        INSERT OR IGNORE INTO news_daily
          (date, symbol, title, source, url, tone)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(a["date"], symbol, a["title"], a["source"], a["url"], a["tone"])
         for a in articles],
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM news_daily").fetchone()[0]
    return after - before


def main() -> int:
    try:
        articles = fetch_articles()
    except Exception as e:  # noqa: BLE001 - fail-open is the point
        print(f"WARN: current-events ingest failed ({e}); continuing without news")
        if FAIL_OPEN:
            return 0
        raise

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        added = store(con, SYMBOL, articles)
        print(f"news_daily: fetched {len(articles)}, +{added} new rows for {SYMBOL}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
