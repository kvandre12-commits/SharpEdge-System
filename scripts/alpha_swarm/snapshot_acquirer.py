#!/usr/bin/env python3
"""Compile immutable provider captures for the paper-only Alpha Swarm pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    slots_by_id,
    validate_manifest,
)
from scripts.alpha_swarm.data_steward import SNAPSHOT_SCHEMA  # noqa: E402
from scripts.alpha_swarm.options_expression_agent import (  # noqa: E402
    OPTION_SNAPSHOT_SCHEMA,
)

ACQUISITION_SCHEMA = "sharpedge.alpha_swarm.provider_capture.v1"
MIN_FEATURE_BARS = 16
RECENT_VOLUME_BARS = 5


def payload_sha256(payload: Any) -> str:
    """Hash canonical provider evidence, not a lossy transformed subset."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_once(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable artifact; replacement is deliberately forbidden."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def _slot(manifest: dict[str, Any], slot_id: str) -> dict[str, Any]:
    validate_manifest(manifest)
    slots = slots_by_id(manifest)
    if slot_id not in slots:
        raise ValueError("slot_id is not present in the locked manifest")
    return slots[slot_id]


def _capture_identity(
    capture: dict[str, Any],
    *,
    symbol: str,
    session_date: str,
    captured_at: datetime,
    name: str,
) -> tuple[datetime, str]:
    if capture.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError(f"{name} schema must be {ACQUISITION_SCHEMA}")
    if capture.get("symbol") != symbol:
        raise ValueError(f"{name} symbol does not match the locked slot")
    if capture.get("session_date") != session_date:
        raise ValueError(f"{name} session_date does not match the locked slot")
    provider = str(capture.get("provider") or "").strip()
    source_ref = str(capture.get("source_ref") or "").strip()
    if not provider or not source_ref:
        raise ValueError(f"{name} provider and source_ref are required")
    latest = parse_timestamp(capture.get("latest_data_ts"), f"{name}.latest_data_ts")
    if latest > captured_at:
        raise ValueError(f"{name} latest_data_ts cannot be after captured_at")
    return latest, source_ref


def _number(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _validated_bars(
    capture: dict[str, Any], *, captured_at: datetime, session_date: str
) -> list[dict[str, Any]]:
    bars = capture.get("bars")
    if not isinstance(bars, list) or len(bars) < MIN_FEATURE_BARS:
        raise ValueError(f"price capture requires at least {MIN_FEATURE_BARS} bars")
    clean: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, bar in enumerate(bars):
        timestamp = parse_timestamp(bar.get("timestamp"), f"bars[{index}].timestamp")
        if timestamp > captured_at:
            raise ValueError("price bars cannot be future-dated")
        if timestamp.date().isoformat() != session_date:
            raise ValueError("price bars must belong to the locked session")
        if timestamp in seen:
            raise ValueError("price bar timestamps must be unique")
        seen.add(timestamp)
        row = {
            "timestamp": timestamp,
            "open": _number(bar.get("open"), f"bars[{index}].open"),
            "high": _number(bar.get("high"), f"bars[{index}].high"),
            "low": _number(bar.get("low"), f"bars[{index}].low"),
            "close": _number(bar.get("close"), f"bars[{index}].close"),
            "volume": _number(bar.get("volume"), f"bars[{index}].volume"),
        }
        if row["close"] <= 0 or row["volume"] < 0:
            raise ValueError(
                "price bars require positive closes and non-negative volume"
            )
        clean.append(row)
    clean.sort(key=lambda row: row["timestamp"])
    return clean


def _neutral_features(bars: list[dict[str, Any]]) -> dict[str, float]:
    weighted = sum(
        ((bar["high"] + bar["low"] + bar["close"]) / 3.0) * bar["volume"]
        for bar in bars
    )
    total_volume = sum(bar["volume"] for bar in bars)
    if total_volume <= 0:
        raise ValueError("price capture needs positive cumulative volume")
    spot = bars[-1]["close"]
    vwap = weighted / total_volume
    anchor = bars[-16]["close"]
    recent = bars[-RECENT_VOLUME_BARS:]
    baseline = bars[:-RECENT_VOLUME_BARS]
    recent_mean = sum(bar["volume"] for bar in recent) / len(recent)
    baseline_mean = sum(bar["volume"] for bar in baseline) / len(baseline)
    volume_ratio = recent_mean / baseline_mean if baseline_mean > 0 else 0.0
    return {
        "spot": round(spot, 6),
        "vwap": round(vwap, 6),
        "vs_vwap_pct": round((spot / vwap - 1.0) * 100.0, 6),
        "momentum_15m_pct": round((spot / anchor - 1.0) * 100.0, 6),
        "volume_ratio": round(volume_ratio, 6),
    }


def build_research_snapshot(
    manifest: dict[str, Any],
    *,
    slot_id: str,
    captured_at: datetime,
    price_capture: dict[str, Any],
    options_capture: dict[str, Any],
) -> dict[str, Any]:
    """Transform two raw captures without deciding eligibility or direction."""
    slot = _slot(manifest, slot_id)
    price_latest, price_ref = _capture_identity(
        price_capture,
        symbol=slot["symbol"],
        session_date=slot["session_date"],
        captured_at=captured_at,
        name="price_capture",
    )
    option_latest, option_ref = _capture_identity(
        options_capture,
        symbol=slot["symbol"],
        session_date=slot["session_date"],
        captured_at=captured_at,
        name="options_capture",
    )
    bars = _validated_bars(
        price_capture, captured_at=captured_at, session_date=slot["session_date"]
    )
    contracts = options_capture.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("options_capture.contracts must be a list")
    option_spot = _number(options_capture.get("spot"), "options_capture.spot")
    if option_spot <= 0:
        raise ValueError("options_capture.spot must be positive")
    features = _neutral_features(bars)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": captured_at.isoformat(),
        "feature_available_ts": max(price_latest, option_latest).isoformat(),
        "features": features,
        "price_source": {
            "provider": price_capture["provider"],
            "source_sha256": payload_sha256(price_capture),
            "latest_data_ts": price_latest.isoformat(),
            "bar_count": len(bars),
            "spot": features["spot"],
        },
        "options_source": {
            "provider": options_capture["provider"],
            "source_sha256": payload_sha256(options_capture),
            "latest_data_ts": option_latest.isoformat(),
            "contract_count": len(contracts),
            "spot": option_spot,
        },
        "source_refs": [price_ref, option_ref],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def build_option_snapshot(
    manifest: dict[str, Any],
    *,
    slot_id: str,
    captured_at: datetime,
    options_capture: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a chain capture for Phase 4 without selecting a contract."""
    slot = _slot(manifest, slot_id)
    latest, source_ref = _capture_identity(
        options_capture,
        symbol=slot["symbol"],
        session_date=slot["session_date"],
        captured_at=captured_at,
        name="options_capture",
    )
    contracts = options_capture.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("options_capture.contracts must be a list")
    normalized = []
    for index, contract in enumerate(contracts):
        normalized.append(
            {
                "contract_symbol": str(contract.get("contract_symbol") or ""),
                "option_type": str(contract.get("option_type") or "").lower(),
                "expiration": str(contract.get("expiration") or ""),
                "strike": _number(contract.get("strike"), f"contracts[{index}].strike"),
                "bid": _number(contract.get("bid"), f"contracts[{index}].bid"),
                "ask": _number(contract.get("ask"), f"contracts[{index}].ask"),
                "quote_ts": parse_timestamp(
                    contract.get("quote_ts"), f"contracts[{index}].quote_ts"
                ).isoformat(),
                "open_interest": int(
                    _number(contract.get("open_interest"), "open_interest")
                ),
                "volume": int(_number(contract.get("volume"), "volume")),
            }
        )
    return {
        "schema": OPTION_SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": captured_at.isoformat(),
        "spot": _number(options_capture.get("spot"), "options_capture.spot"),
        "source": {
            "provider": options_capture["provider"],
            "source_sha256": payload_sha256(options_capture),
            "source_ref": source_ref,
            "latest_data_ts": latest.isoformat(),
        },
        "contracts": normalized,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def acquire_research_snapshot(
    manifest: dict[str, Any],
    *,
    slot_id: str,
    captured_at: datetime,
    price_fetcher: Callable[[str], dict[str, Any]],
    options_fetcher: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Inject fetchers so provider I/O stays separate and tests stay offline."""
    slot = _slot(manifest, slot_id)
    price = price_fetcher(slot["symbol"])
    options = options_fetcher(slot["symbol"])
    return build_research_snapshot(
        manifest,
        slot_id=slot_id,
        captured_at=captured_at,
        price_capture=price,
        options_capture=options,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--options-capture", required=True, type=Path)
    parser.add_argument("--price-capture", type=Path)
    parser.add_argument("--kind", choices=("research", "option"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit("capture compilation requires --no-network")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    options = json.loads(args.options_capture.read_text(encoding="utf-8"))
    captured_at = parse_timestamp(args.captured_at, "captured_at")
    if args.kind == "research":
        if args.price_capture is None:
            raise SystemExit("research capture requires --price-capture")
        price = json.loads(args.price_capture.read_text(encoding="utf-8"))
        artifact = build_research_snapshot(
            manifest,
            slot_id=args.slot_id,
            captured_at=captured_at,
            price_capture=price,
            options_capture=options,
        )
    else:
        artifact = build_option_snapshot(
            manifest,
            slot_id=args.slot_id,
            captured_at=captured_at,
            options_capture=options,
        )
    write_once(args.output, artifact)
    print(json.dumps({"schema": artifact["schema"], "slot_id": artifact["slot_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
