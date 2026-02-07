#!/usr/bin/env python3
import os
import sqlite3
import requests
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

if not WEBHOOK:
    raise RuntimeError("Set DISCORD_WEBHOOK_URL environment variable")

def latest_execution_row(conn):
    df = pd.read_sql_query(
        """
        SELECT *
        FROM execution_state_daily
        WHERE symbol = ?
        ORDER BY session_date DESC
        LIMIT 1
        """,
        conn,
        params=(SYMBOL,),
    )
    if df.empty:
        raise RuntimeError("No execution_state_daily rows found")
    return df.iloc[0].to_dict()

def build_message(r):
    bias = r.get("final_bias", "—")
    score = round(float(r.get("execution_score", 0)), 1)
    trend = round(float(r.get("prob_trend_fused", 0)), 2)
    dealer = r.get("dealer_state_hint", "—")
    gamma = round(float(r.get("gamma_proxy", 0)), 2)
    wall = r.get("wall_strike", "—")
    dist = round(float(r.get("dist_to_wall_pct", 0)), 2)
    comp = r.get("compression_flag", "—")
    cluster = round(float(r.get("cluster_score", 0)), 3)
    date = r.get("session_date", "—")

    # Emoji by bias
    if "EXPANSION" in bias:
        icon = "🚀"
    elif "PIN" in bias:
        icon = "🧲"
    elif "WHIP" in bias:
        icon = "⚠️"
    else:
        icon = "➖"

    return f"""
{icon} **SPY TRADE CARD**

📅 **Date:** {date}  
🎯 **Bias:** **{bias}**  
📊 **Execution Score:** **{score}/100**

---

📈 **Trend Probability (fused):** {trend}  
🏦 **Dealer State:** {dealer}  
🧮 **Gamma Proxy:** {gamma}

🧲 **Wall Strike:** {wall}  
📏 **Distance → Wall:** {dist}%

---

🌪 **Compression:** {comp}  
🧊 **Cluster Score:** {cluster}

---

_This is decision support — not financial advice._
""".strip()

def send(msg):
    requests.post(WEBHOOK, json={"content": msg}, timeout=10)

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        row = latest_execution_row(conn)
        msg = build_message(row)
        send(msg)
        print("✅ Trade card sent to Discord")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
