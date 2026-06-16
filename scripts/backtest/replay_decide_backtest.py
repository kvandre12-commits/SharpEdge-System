"""Faithful backtest of the Bridge ``decide()`` gates on real history.

Reconstructs a per-session intraday signal (the exact fields ``decide()`` reads)
from ``spy_bars_15m`` + ``options_chain_snapshots``, grades each through the
Bridge backtest engine, and prints per-gate-state edge.

Design honesty:
- Faithful signals need options data, so coverage = the bars/options OVERLAP
  (~87 sessions, 2026). We do NOT fake the Jan-2025 bars-only window.
- Decision point is intraday (default ~40% through the session ≈ late morning),
  matching a same-day 0DTE momentum entry. Forward return = decision -> close.
- gamma_regime uses a Gamma-Exposure proxy; when gamma is missing we leave it
  None so ``decide()`` conservatively stands down (no fabricated trades).

This script READS only. It writes one CSV + prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path
from statistics import median

# Make the Bridge engine importable without installing it.
_BRIDGE_SRC = Path(os.path.expanduser("~/SharpEdge-Robinhood-Bridge/src"))
if _BRIDGE_SRC.exists():
    sys.path.insert(0, str(_BRIDGE_SRC))

from sharpedge_robinhood_bridge.backtest import run_backtest  # noqa: E402

DB = Path(os.path.expanduser("~/SharpEdge-System/data/spy_truth.db"))
OUT = Path(os.path.expanduser("~/SharpEdge-System/outputs/decide_backtest.csv"))
DECISION_FRACTION = 0.40  # how far through the session we "decide"


def _sessions_with_both(cur: sqlite3.Cursor) -> list[str]:
    rows = cur.execute(
        "SELECT DISTINCT b.session_date FROM spy_bars_15m b "
        "JOIN options_chain_snapshots o ON o.session_date = b.session_date "
        "ORDER BY b.session_date"
    ).fetchall()
    return [r[0] for r in rows]


def _session_bars(cur: sqlite3.Cursor, day: str) -> list[dict]:
    rows = cur.execute(
        "SELECT ts, open, high, low, close, volume FROM spy_bars_15m "
        "WHERE session_date=? ORDER BY ts",
        (day,),
    ).fetchall()
    return [
        {"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] or 0.0}
        for r in rows
    ]


def _walls_and_regime(cur: sqlite3.Cursor, day: str, spot: float) -> dict:
    """Nearest-expiry walls + gamma regime, matching cockpit/gamma.py exactly.

    Live convention (read_options / gamma_profile):
      call_wall = max-OI strike AT-OR-ABOVE spot; put_wall = max-OI AT-OR-BELOW.
      net_gamma = sum(call_gamma*call_oi - put_gamma*put_oi); missing gamma -> 0.
      regime = 'positive' if net_gamma >= 0 else 'negative' (never None, matches live).
    """
    nearest = cur.execute(
        "SELECT MIN(dte) FROM options_chain_snapshots WHERE session_date=? AND dte>=0",
        (day,),
    ).fetchone()[0]
    if nearest is None:
        return {"call_wall": None, "put_wall": None, "gamma_regime": "positive", "exp": None}
    rows = cur.execute(
        "SELECT expiry_date, strike, call_oi, put_oi, call_gamma, put_gamma "
        "FROM options_chain_snapshots WHERE session_date=? AND dte=?",
        (day, nearest),
    ).fetchall()
    exp = rows[0][0] if rows else None

    calls_above = [(r[1], r[2] or 0) for r in rows if r[1] >= spot]
    puts_below = [(r[1], r[3] or 0) for r in rows if r[1] <= spot]
    call_wall = max(calls_above, key=lambda x: x[1])[0] if calls_above else None
    put_wall = max(puts_below, key=lambda x: x[1])[0] if puts_below else None

    net_gamma = 0.0
    for _exp, _strike, c_oi, p_oi, c_g, p_g in rows:
        net_gamma += (c_g or 0.0) * (c_oi or 0.0) - (p_g or 0.0) * (p_oi or 0.0)
    regime = "positive" if net_gamma >= 0 else "negative"
    return {"call_wall": call_wall, "put_wall": put_wall,
            "gamma_regime": regime, "exp": exp}


def build_signal(cur: sqlite3.Cursor, day: str) -> tuple[dict, float] | None:
    bars = _session_bars(cur, day)
    if len(bars) < 5:
        return None
    idx = max(1, int(len(bars) * DECISION_FRACTION))
    sofar = bars[: idx + 1]  # bars available at decision time (no lookahead)
    closes = [b["close"] for b in sofar]
    vols = [b["volume"] for b in sofar]
    spot = closes[-1]

    # VWAP exactly as live: sum(close*vol)/sum(vol)
    cum_pv = sum(b["close"] * b["volume"] for b in sofar)
    cum_v = sum(vols) or 1
    vwap = cum_pv / cum_v
    vs_vwap = (spot - vwap) / vwap * 100 if vwap else 0.0

    # momentum: look back min(15, n-1) BARS (live definition)
    look = min(15, len(closes) - 1)
    mom15 = (spot / closes[-1 - look] - 1) * 100 if look else 0.0

    # volume confirmation: mean(last 5 bars) / median(all bars so far)
    recent = vols[-5:]
    med = median(vols) or 1
    vol_mult = (sum(recent) / len(recent)) / med

    opt = _walls_and_regime(cur, day, spot)
    close = bars[-1]["close"]
    fwd_return_pct = (close / spot - 1.0) * 100.0 if spot else 0.0

    signal = {
        "session": day,
        "symbol": "SPY",
        "spot": round(spot, 2),
        "vwap": round(vwap, 2),
        "vs_vwap": round(vs_vwap, 3),
        "mom15": round(mom15, 3),
        "vol_mult": round(vol_mult, 2),
        "atm_iv": 0.15,  # not stored historically; nominal for premium estimate only
        **opt,
    }
    return signal, round(fwd_return_pct, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--json", action="store_true", help="print summary as JSON")
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    cur = c.cursor()
    sessions = _sessions_with_both(cur)

    records: list[tuple[dict, float]] = []
    for day in sessions:
        built = build_signal(cur, day)
        if built:
            records.append(built)

    result = run_backtest(records)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "action", "reason", "direction", "fwd_return_pct", "signed_return_pct"])
        for t in result.trades:
            w.writerow([t.session, t.action, t.reason, t.direction or "",
                        t.fwd_return_pct, t.signed_return_pct if t.signed_return_pct is not None else ""])

    summary = result.summary()
    summary["coverage_sessions"] = len(records)
    summary["date_range"] = [sessions[0], sessions[-1]] if sessions else []

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Coverage: {len(records)} sessions  {summary['date_range']}")
        print(f"Signals: {summary['n_signals']}  Trades: {summary['n_trades']}  "
              f"Win rate: {summary['win_rate']}  Avg signed ret: {summary['avg_signed_return_pct']}")
        print("\nBy gate state (reason -> n / win_rate / avg signed return %):")
        for reason, s in summary["by_reason"].items():
            wr = f"{s['win_rate']:.2f}" if s["win_rate"] is not None else "  -"
            av = f"{s['avg_signed_return_pct']:+.3f}" if s["avg_signed_return_pct"] is not None else "  -"
            print(f"  [{s['n']:3d}] wr={wr} avg={av}  {reason}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
