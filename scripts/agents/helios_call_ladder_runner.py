#!/usr/bin/env python3
"""Publish one idempotent SPY/NVDA Helios call-ladder lead from noon–12:30 ET."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
COCKPIT = ROOT / "cockpit"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(COCKPIT) not in sys.path:
    sys.path.insert(0, str(COCKPIT))

from market_data_sources import fetch_yahoo_regular_session_chart_rows

from scripts.alpha_swarm.contracts import canonical_json, payload_sha256
from scripts.alpha_swarm.helios_call_ladder import build_lead

NY = ZoneInfo("America/New_York")
OUTDIR = ROOT / "outputs" / "helios_call_ladder"
SNAPSHOT_DIR = ROOT / "outputs" / "nerv_helios_call_ladder"
GAP_WATCH = ROOT / "outputs" / "robinhood_gap_watch.json"
STATE_PATH = OUTDIR / "cycle_state.json"
LATEST_PATH = OUTDIR / "latest.json"
WALLET_HANDOFF_PATH = OUTDIR / "wallet_buyer_handoff.json"
BRIDGE_ROOT = Path(
    os.getenv(
        "SHARPEDGE_ROBINHOOD_BRIDGE_ROOT",
        str(ROOT.parent / "SharpEdge-Robinhood-Bridge"),
    )
)
SYMBOLS = ("SPY", "NVDA")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _due(now: datetime) -> bool:
    local = now.astimezone(NY)
    return local.weekday() < 5 and time(12, 0) <= local.time() <= time(12, 30)


def _features(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if len(rows) < 16:
        return None
    closes = [float(row["close"]) for row in rows]
    volumes = [int(row.get("volume") or 0) for row in rows]
    total_volume = sum(volumes)
    if not total_volume or not closes[-16]:
        return None
    vwap = sum(
        ((float(row["high"]) + float(row["low"]) + float(row["close"])) / 3.0) * volume
        for row, volume in zip(rows, volumes)
    ) / total_volume
    baseline = statistics.median(volumes) or 1.0
    return {
        "spot": round(closes[-1], 4),
        "vwap": round(vwap, 4),
        "vs_vwap_pct": round((closes[-1] / vwap - 1.0) * 100.0, 4),
        "momentum_15m_pct": round((closes[-1] / closes[-16] - 1.0) * 100.0, 4),
        "volume_ratio": round(statistics.mean(volumes[-5:]) / baseline, 4),
    }


def _nearest_targets(gap_watch: dict[str, Any]) -> dict[str, float | None]:
    targets: dict[str, float | None] = {symbol: None for symbol in SYMBOLS}
    for watch in gap_watch.get("watches") or []:
        symbol = str(watch.get("symbol") or "").upper()
        if symbol in targets:
            targets[symbol] = (watch.get("refill_structure") or {}).get("nearest_target")
    return targets


def _refresh_nerv() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "nerv_free_data_adapter.py"),
        "--symbols",
        ",".join(SYMBOLS),
        "--max-expirations",
        "4",
        "--board-limit",
        "0",
        "--output-dir",
        str(SNAPSHOT_DIR),
        "--retention-hours",
        "24",
        "--skip-panel-refresh",
        "--skip-curator-refresh",
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=120)


def _wallet_cycle_state(state: dict[str, Any], session_date: str) -> str:
    for item in (state.get("symbols") or {}).values():
        current = str(item.get("state") or "IDLE")
        if current in {"OPEN_POSITION", "SELLER_MANAGING", "EXIT_PENDING_CONFIRM"}:
            return current
        if current == "ENTRY_PROPOSED" and item.get("session_date") == session_date:
            return current
    return "IDLE"


def _snapshot_is_fresh(path: Path, now: datetime, max_age_minutes: float = 5.0) -> bool:
    summary = (_read_json(path).get("summary") or {}) if path.exists() else {}
    try:
        fetched = datetime.fromisoformat(str(summary.get("fetch_timestamp") or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    age = (now.astimezone(UTC) - fetched.astimezone(UTC)).total_seconds() / 60.0
    return 0 <= age <= max_age_minutes


def _apply_global_wallet(leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        (float(item.get("score") or 0.0), lead["symbol"], item)
        for lead in leads
        if lead.get("aggregate_decision") == "approve"
        for item in lead.get("ladder") or []
    ]
    candidates.sort(key=lambda row: (-row[0], row[1], str(row[2].get("contract_symbol"))))
    selected: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in SYMBOLS}
    spent = 0.0
    for _score, symbol, item in candidates:
        cost = float(item.get("max_debit_dollars") or 0.0)
        if cost > 0 and spent + cost <= 250.0:
            selected[symbol].append(item)
            spent += cost

    for lead in leads:
        symbol = lead["symbol"]
        originally_approved = lead.get("aggregate_decision") == "approve"
        lead["ladder"] = selected.get(symbol, [])
        debit = round(sum(float(item["max_debit_dollars"]) for item in lead["ladder"]), 2)
        lead["wallet"] = {
            "global_cap_dollars": 250.0,
            "proposed_debit_dollars": debit,
            "global_proposed_debit_dollars": round(spent, 2),
            "global_remaining_dollars": round(250.0 - spent, 2),
        }
        if originally_approved and not lead["ladder"]:
            lead["aggregate_decision"] = "abstain"
            lead["entry_handoff"]["state"] = "STAND_DOWN"
            lead["reviews"].append(
                {
                    "role": "global_wallet_allocator",
                    "decision": "abstain",
                    "reasons": ["global_wallet_allocated_elsewhere"],
                    "authoritative": False,
                    "execution_permitted": False,
                }
            )
        unsigned = dict(lead)
        unsigned.pop("lead_id", None)
        lead["lead_id"] = payload_sha256(unsigned)
    return leads


def build_run(now: datetime, *, refresh_nerv: bool = True) -> dict[str, Any]:
    local = now.astimezone(NY)
    session_date = local.date().isoformat()
    snapshot_path = SNAPSHOT_DIR / "nerv_options_snapshot.json"
    if refresh_nerv and not _snapshot_is_fresh(snapshot_path, now):
        _refresh_nerv()
    snapshot = _read_json(snapshot_path)
    gap_watch = _read_json(GAP_WATCH)
    state = _read_json(STATE_PATH)
    targets = _nearest_targets(gap_watch)
    quotes = snapshot.get("quotes") or []
    snapshot_summary = snapshot.get("summary") or {}
    fetch_timestamp = str(snapshot_summary.get("fetch_timestamp") or "")
    data_issues: list[str] = []
    try:
        fetched = datetime.fromisoformat(fetch_timestamp.replace("Z", "+00:00"))
        age_minutes = (now.astimezone(UTC) - fetched.astimezone(UTC)).total_seconds() / 60.0
        if age_minutes < 0 or age_minutes > 20:
            data_issues.append("stale_nerv_snapshot")
    except ValueError:
        data_issues.append("missing_nerv_fetch_timestamp")

    leads = []
    feature_packets: dict[str, Any] = {}
    cycle_state = _wallet_cycle_state(state, session_date)
    for symbol in SYMBOLS:
        rows, source = fetch_yahoo_regular_session_chart_rows(symbol, timeout=15)
        features = _features(rows)
        symbol_issues = list(data_issues)
        if source.get("session_date") != session_date:
            symbol_issues.append("price_session_mismatch")
        if features is None:
            symbol_issues.append("insufficient_intraday_bars")
            features = {}
        feature_packets[symbol] = {"features": features, "source": source}
        symbol_quotes = [
            quote for quote in quotes if str(quote.get("underlying") or "").upper() == symbol
        ]
        source_hashes = {
            "nerv_snapshot": _sha256(snapshot_path) if snapshot_path.exists() else "",
            "gap_watch": _sha256(GAP_WATCH) if GAP_WATCH.exists() else "",
            "feature_packet": payload_sha256(feature_packets[symbol]),
        }
        leads.append(
            build_lead(
                symbol=symbol,
                now=now,
                features=features,
                quotes=symbol_quotes,
                nearest_target=targets.get(symbol),
                source_hashes=source_hashes,
                cycle_state=cycle_state,
                data_issues=sorted(set(symbol_issues)),
            )
        )

    leads = _apply_global_wallet(leads)
    payload = {
        "schema": "sharpedge.helios_call_ladder_run.v1",
        "generated_at": now.isoformat(),
        "session_date": session_date,
        "timezone": "America/New_York",
        "leads": leads,
        "authority": {
            "authoritative": False,
            "execution_permitted": False,
            "broker_action_allowed": False,
        },
    }
    payload["run_id"] = payload_sha256(payload)
    return payload


def _update_state(payload: dict[str, Any]) -> None:
    state = _read_json(STATE_PATH)
    symbols = state.setdefault("symbols", {})
    total_reserved = 0.0
    for lead in payload["leads"]:
        symbol = lead["symbol"]
        existing = symbols.get(symbol) or {}
        if lead["aggregate_decision"] == "approve":
            debit = float((lead.get("wallet") or {}).get("proposed_debit_dollars") or 0.0)
            total_reserved += debit
            symbols[symbol] = {
                "state": "ENTRY_PROPOSED",
                "session_date": payload["session_date"],
                "trigger_id": lead["lead_id"],
                "lead_id": lead["lead_id"],
                "reserved_dollars": debit,
            }
        else:
            symbols[symbol] = {
                **existing,
                "state": existing.get("state", "IDLE"),
                "last_evaluated_session": payload["session_date"],
                "last_decision": lead["aggregate_decision"],
            }
    state["wallet"] = {
        "cap_dollars": 250.0,
        "reserved_dollars": round(total_reserved, 2),
        "available_dollars": round(250.0 - total_reserved, 2),
    }
    state["updated_at"] = payload["generated_at"]
    _atomic_json(STATE_PATH, state)


def _publish_wallet_handoff(lead_path: Path) -> None:
    source_root = BRIDGE_ROOT / "src"
    if not source_root.exists():
        _atomic_json(
            WALLET_HANDOFF_PATH,
            {
                "schema": "sharpedge.helios_wallet_buyer_handoff.v1",
                "status": "blocked",
                "blockers": ["robinhood_bridge_source_unavailable"],
                "source_lead_run": str(lead_path),
            },
        )
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    command = [
        sys.executable,
        "-m",
        "sharpedge_robinhood_bridge.helios_wallet_buyer",
        "--lead-run",
        str(lead_path),
        "--output",
        str(WALLET_HANDOFF_PATH),
    ]
    completed = subprocess.run(
        command,
        cwd=BRIDGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        _atomic_json(
            WALLET_HANDOFF_PATH,
            {
                "schema": "sharpedge.helios_wallet_buyer_handoff.v1",
                "status": "blocked",
                "blockers": ["wallet_buyer_adapter_failed"],
                "source_lead_run": str(lead_path),
                "error": completed.stderr.strip()[:500],
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="run outside the scheduler for bounded validation")
    parser.add_argument("--no-refresh", action="store_true", help="reuse the current NERV snapshot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(UTC)
    if not args.force and not _due(now):
        return 0
    local = now.astimezone(NY)
    daily_path = OUTDIR / local.date().isoformat() / "lead_1230.json"
    state = _read_json(STATE_PATH)
    if _wallet_cycle_state(state, local.date().isoformat()) != "IDLE" and not args.force:
        return 0
    payload = build_run(now, refresh_nerv=not args.no_refresh)
    _atomic_json(daily_path, payload)
    _atomic_json(LATEST_PATH, payload)
    _update_state(payload)
    _publish_wallet_handoff(daily_path)
    print(canonical_json({"run_id": payload["run_id"], "decisions": {lead["symbol"]: lead["aggregate_decision"] for lead in payload["leads"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
