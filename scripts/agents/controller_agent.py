#!/usr/bin/env python3
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUTDIR / "agent_controller_decision.json"
OUT_TXT = OUTDIR / "agent_controller_prompt.txt"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (name,)).fetchone() is not None


def safe_read_sql(con: sqlite3.Connection, query: str, params=()):
    try:
        return pd.read_sql_query(query, con, params=params)
    except Exception:
        return pd.DataFrame()


def get_latest_signal(con: sqlite3.Connection) -> dict:
    if not table_exists(con, "signals_daily"):
        return {}

    df = safe_read_sql(
        con,
        """
        SELECT *
        FROM signals_daily
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (SYMBOL,),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_latest_regime(con: sqlite3.Connection) -> dict:
    if not table_exists(con, "regime_daily"):
        return {}

    df = safe_read_sql(
        con,
        """
        SELECT *
        FROM regime_daily
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (SYMBOL,),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_latest_open_regime(con: sqlite3.Connection) -> dict:
    if not table_exists(con, "open_resolution_regime"):
        return {}

    df = safe_read_sql(
        con,
        """
        SELECT *
        FROM open_resolution_regime
        WHERE underlying = ?
        ORDER BY session_date DESC
        LIMIT 1
        """,
        (SYMBOL,),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_latest_positioning(con: sqlite3.Connection) -> dict:
    if not table_exists(con, "options_positioning_metrics"):
        return {}

    df = safe_read_sql(
        con,
        """
        SELECT *
        FROM options_positioning_metrics
        WHERE underlying = ?
        ORDER BY session_date DESC, snapshot_ts DESC
        LIMIT 1
        """,
        (SYMBOL,),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_latest_trade_card_text() -> str:
    latest_signal = OUTDIR / "latest_signal_strength.csv"
    if latest_signal.exists():
        try:
            df = pd.read_csv(latest_signal)
            return df.to_csv(index=False)
        except Exception:
            return ""
    return ""


def normalize_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def normalize_dict(d: dict) -> dict:
    return {k: normalize_value(v) for k, v in d.items()}


def build_prompt(signal, regime, open_regime, positioning, trade_card_text) -> str:
    system = """
You are a risk controller for a SPY trading pipeline.

Your job is NOT to create a trade.
Your job is to decide whether the finished pipeline output should be published to Discord.

You must return ONLY valid JSON with this exact schema:
{
  "decision": "post" or "hold",
  "confidence": 0.0 to 1.0,
  "summary": "short summary",
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "risk_flags": ["flag 1", "flag 2"]
}

Use conservative logic:
- choose "hold" if evidence is mixed, incomplete, contradictory, or weak
- choose "hold" if key fields are missing
- choose "post" only if the signal/regime/positioning state is coherent
- keep reasons short and concrete
""".strip()

    user = {
        "latest_signal": normalize_dict(signal),
        "latest_regime": normalize_dict(regime),
        "latest_open_regime": normalize_dict(open_regime),
        "latest_options_positioning": normalize_dict(positioning),
        "latest_trade_card_artifact": trade_card_text,
    }

    return f"{system}\n\nDATA:\n{json.dumps(user, indent=2)}"


def call_gemini(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        return {
            "decision": "hold",
            "confidence": 0.0,
            "summary": "Missing GEMINI_API_KEY",
            "reasons": ["No Gemini key available"],
            "risk_flags": ["agent_not_configured"],
        }

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json",
        },
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

    if not text:
        raise RuntimeError("Gemini returned empty response")

    parsed = json.loads(text)

    # defensive cleanup
    decision = str(parsed.get("decision", "hold")).lower()
    if decision not in {"post", "hold"}:
        decision = "hold"

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.0

    return {
        "decision": decision,
        "confidence": confidence,
        "summary": str(parsed.get("summary", "")),
        "reasons": list(parsed.get("reasons", []))[:6],
        "risk_flags": list(parsed.get("risk_flags", []))[:6],
    }


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        signal = get_latest_signal(con)
        regime = get_latest_regime(con)
        open_regime = get_latest_open_regime(con)
        positioning = get_latest_positioning(con)
    finally:
        con.close()

    prompt = build_prompt(
        signal=signal,
        regime=regime,
        open_regime=open_regime,
        positioning=positioning,
        trade_card_text=get_latest_trade_card_text(),
    )

    OUT_TXT.write_text(prompt, encoding="utf-8")

    try:
        result = call_gemini(prompt)
    except Exception as e:
        result = {
            "decision": "hold",
            "confidence": 0.0,
            "summary": f"Controller failed: {type(e).__name__}",
            "reasons": [str(e)],
            "risk_flags": ["controller_error"],
        }

    result["symbol"] = SYMBOL
    result["ts_utc"] = datetime.now(timezone.utc).isoformat()

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"controller_decision={result['decision']}")


if __name__ == "__main__":
    main()
