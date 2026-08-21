#!/usr/bin/env python3
"""Deterministic paper-only options expression for locked alpha-swarm candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
    validate_candidate,
)
from scripts.alpha_swarm.hypothesis_researcher import (  # noqa: E402
    PUBLICATION_SCHEMA as PHASE3_SCHEMA,
)

OPTION_SNAPSHOT_SCHEMA = "sharpedge.alpha_swarm.option_chain_snapshot.v1"
PUBLICATION_SCHEMA = "sharpedge.alpha_swarm.options_expression_publication.v1"
EXPRESSION_SCHEMA = "sharpedge.alpha_swarm.options_expression.v1"
RULE_ID = "adjacent_atm_debit_spread_v1"
RULE_VERSION = "1.0.0"
MIN_DTE = 7
MAX_DTE = 21
TARGET_DTE = 14
MAX_QUOTE_AGE_MINUTES = 20.0
MAX_QUOTE_WIDTH_PCT = 25.0
MIN_OPEN_INTEREST = 100
MIN_VOLUME = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "utility",
        "performance",
        "score",
        "alpha_score",
        "rank",
        "ret_1d",
        "return_prediction_to_exit",
        "broker",
        "route",
        "order_id",
    }
)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _walk_forbidden(value: Any, *, path: str = "input") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_INPUT_FIELDS:
                failures.append(f"forbidden field {path}.{key}")
            failures.extend(_walk_forbidden(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden(child, path=f"{path}[{index}]"))
    return failures


def _phase3_source_sha256() -> str:
    return _source_sha256(Path(__file__).with_name("hypothesis_researcher.py"))


def _validate_phase3(publication: dict[str, Any]) -> None:
    if publication.get("schema") != PHASE3_SCHEMA:
        raise ValueError(f"Phase 3 schema must be {PHASE3_SCHEMA}")
    if publication.get("paper_only") is not True:
        raise ValueError("Phase 3 publication must remain paper_only")
    if publication.get("authoritative") is not False:
        raise ValueError("Phase 3 publication must remain non-authoritative")
    if publication.get("execution_permitted") is not False:
        raise ValueError("Phase 3 publication must remain non-executable")
    if publication.get("option_selection_allowed") is not False:
        raise ValueError("Phase 3 publication must not select an option")
    if publication.get("researcher_source_sha256") != _phase3_source_sha256():
        raise ValueError(
            "Phase 3 researcher source SHA256 does not match current source"
        )
    forbidden = _walk_forbidden(publication, path="phase3")
    if forbidden:
        raise ValueError(f"Phase 3 publication contains forbidden fields: {forbidden}")


def _base(publication: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "schema": PUBLICATION_SCHEMA,
        "run_id": publication.get("run_id"),
        "manifest_sha256": publication.get("manifest_sha256"),
        "evaluator_source_sha256": publication.get("evaluator_source_sha256"),
        "phase3_publication_sha256": _artifact_sha256(publication),
        "agent_source_sha256": _source_sha256(Path(__file__)),
        "slot_id": publication.get("slot_id"),
        "session_date": publication.get("session_date"),
        "symbol": publication.get("symbol"),
        "expression_at": now.isoformat(),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
    }


def _upstream_without_candidate(
    publication: dict[str, Any], now: datetime
) -> dict[str, Any] | None:
    state = publication.get("state")
    if state == "not_ready":
        if publication.get("candidate") is not None:
            raise ValueError(
                "not_ready Phase 3 publication must not include a candidate"
            )
        return {
            **_base(publication, now),
            "state": "upstream_not_ready",
            "evaluator_accounting": "none",
            "reason": "Phase 3 candidate is not ready",
            "expression": None,
        }
    if state == "data_rejected":
        if publication.get("candidate") is not None:
            raise ValueError(
                "data_rejected Phase 3 publication must not include a candidate"
            )
        return {
            **_base(publication, now),
            "state": "upstream_data_rejected",
            "evaluator_accounting": "zero_utility_rejection",
            "reason": "Phase 3 rejected upstream evidence",
            "expression": None,
        }
    if state != "candidate_published":
        raise ValueError(
            "Phase 3 state must be not_ready, data_rejected, or candidate_published"
        )
    return None


def _validate_candidate_identity(
    manifest: dict[str, Any], publication: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = publication.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate_published Phase 3 publication needs one candidate")
    slot = validate_candidate(manifest, candidate)
    for field in ("run_id", "manifest_sha256", "slot_id", "symbol"):
        if candidate.get(field) != publication.get(field):
            raise ValueError(f"Phase 3 candidate {field} does not match publication")
    unsigned = dict(candidate)
    candidate_id = unsigned.pop("candidate_id", None)
    if candidate_id != payload_sha256(unsigned):
        raise ValueError("Phase 3 candidate_id does not match candidate payload")
    return candidate, slot


def _expression_time(candidate: dict[str, Any]) -> datetime:
    return parse_timestamp(candidate["prediction_ts"], "prediction_ts") + timedelta(
        minutes=1
    )


def _validate_snapshot_identity(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    slot: dict[str, Any],
    snapshot: dict[str, Any],
    now: datetime,
) -> tuple[float, list[dict[str, Any]]]:
    forbidden = _walk_forbidden(snapshot, path="option_snapshot")
    if forbidden:
        raise ValueError(f"option snapshot contains forbidden fields: {forbidden}")
    if snapshot.get("schema") != OPTION_SNAPSHOT_SCHEMA:
        raise ValueError(f"option snapshot schema must be {OPTION_SNAPSHOT_SCHEMA}")
    expected = {
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "session_date": slot["session_date"],
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            raise ValueError(f"option snapshot {field} does not match locked candidate")
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("authoritative") is not False
    ):
        raise ValueError("option snapshot must remain paper-only and non-authoritative")
    if snapshot.get("execution_permitted") is not False:
        raise ValueError("option snapshot must remain non-executable")
    captured = parse_timestamp(snapshot.get("captured_at"), "captured_at")
    if captured != now:
        raise ValueError("option snapshot captured_at must equal exact expression time")
    source = snapshot.get("source") or {}
    if not str(source.get("provider") or "").strip():
        raise ValueError("option snapshot source.provider is required")
    if not SHA256_RE.fullmatch(str(source.get("source_sha256") or "")):
        raise ValueError("option snapshot source SHA256 is invalid")
    if not str(source.get("source_ref") or "").strip():
        raise ValueError("option snapshot source_ref is required")
    latest = parse_timestamp(source.get("latest_data_ts"), "source.latest_data_ts")
    age = (captured - latest).total_seconds() / 60.0
    if age < 0 or age > MAX_QUOTE_AGE_MINUTES:
        raise ValueError("option snapshot source is future-dated or stale")
    if latest.date().isoformat() != slot["session_date"]:
        raise ValueError("option snapshot source is outside the locked session")
    spot = _safe_float(snapshot.get("spot"))
    if spot is None or spot <= 0:
        raise ValueError("option snapshot spot must be positive")
    contracts = snapshot.get("contracts")
    if not isinstance(contracts, list) or len(contracts) < 2:
        raise ValueError("option snapshot requires at least two contracts")
    identities: set[tuple[str, str, float]] = set()
    symbols: set[str] = set()
    for contract in contracts:
        symbol = str(contract.get("contract_symbol") or "").strip()
        strike = _safe_float(contract.get("strike"))
        identity = (
            str(contract.get("expiration")),
            str(contract.get("option_type")),
            strike or 0.0,
        )
        if not symbol or symbol in symbols:
            raise ValueError("option contract symbols must be non-empty and unique")
        if identity in identities:
            raise ValueError(
                "option contracts must be unique by expiration/type/strike"
            )
        symbols.add(symbol)
        identities.add(identity)
    return spot, contracts


def _quality_contract(
    contract: dict[str, Any], *, session: date, captured: datetime, option_type: str
) -> dict[str, Any] | None:
    if str(contract.get("option_type") or "").lower() != option_type:
        return None
    try:
        expiration = date.fromisoformat(str(contract.get("expiration")))
    except ValueError:
        return None
    dte = (expiration - session).days
    strike = _safe_float(contract.get("strike"))
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    if not MIN_DTE <= dte <= MAX_DTE or not strike or strike <= 0:
        return None
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2.0
    width_pct = (ask - bid) / midpoint * 100.0 if midpoint else float("inf")
    if width_pct > MAX_QUOTE_WIDTH_PCT:
        return None
    if int(_safe_float(contract.get("open_interest")) or 0) < MIN_OPEN_INTEREST:
        return None
    if int(_safe_float(contract.get("volume")) or 0) < MIN_VOLUME:
        return None
    try:
        quote = parse_timestamp(contract.get("quote_ts"), "quote_ts")
    except ValueError:
        return None
    age = (captured - quote).total_seconds() / 60.0
    if age < 0 or age > MAX_QUOTE_AGE_MINUTES:
        return None
    return {
        "contract_symbol": str(contract["contract_symbol"]),
        "option_type": option_type,
        "expiration": expiration.isoformat(),
        "dte": dte,
        "strike": float(strike),
        "bid": float(bid),
        "ask": float(ask),
        "quote_ts": contract["quote_ts"],
        "open_interest": int(float(contract["open_interest"])),
        "volume": int(float(contract["volume"])),
        "quote_width_pct": round(width_pct, 4),
    }


def _spread_candidates(
    contracts: list[dict[str, Any]],
    *,
    decision: str,
    session: date,
    captured: datetime,
    spot: float,
    risk_cap: float,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    option_type = "call" if decision == "long" else "put"
    quality = [
        normalized
        for contract in contracts
        if (
            normalized := _quality_contract(
                contract, session=session, captured=captured, option_type=option_type
            )
        )
    ]
    by_expiration: dict[str, list[dict[str, Any]]] = {}
    for contract in quality:
        by_expiration.setdefault(contract["expiration"], []).append(contract)
    choices: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for expiration, expiry_contracts in by_expiration.items():
        ordered = sorted(
            expiry_contracts, key=lambda item: (item["strike"], item["contract_symbol"])
        )
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if decision == "long":
            pairs = [
                (ordered[index], ordered[index + 1])
                for index in range(len(ordered) - 1)
                if ordered[index]["strike"] >= spot
            ]
        else:
            pairs = [
                (ordered[index], ordered[index - 1])
                for index in range(1, len(ordered))
                if ordered[index]["strike"] <= spot
            ]
        for long_leg, short_leg in pairs:
            width_dollars = round(
                abs(short_leg["strike"] - long_leg["strike"]) * 100.0, 2
            )
            debit_dollars = round((long_leg["ask"] - short_leg["bid"]) * 100.0, 2)
            max_gain = round(width_dollars - debit_dollars, 2)
            if debit_dollars <= 0 or debit_dollars > risk_cap or max_gain <= 0:
                continue
            expression = {
                "vehicle": "debit_spread",
                "structure": f"{option_type}_debit_spread",
                "quantity": 1,
                "option_type": option_type,
                "expiration": expiration,
                "dte": long_leg["dte"],
                "long_leg": long_leg,
                "short_leg": short_leg,
                "entry_method": "buy_ask_sell_bid",
                "entry_debit_dollars": debit_dollars,
                "max_loss_dollars": debit_dollars,
                "max_gain_dollars": max_gain,
                "spread_width_dollars": width_dollars,
            }
            key = (
                abs(long_leg["dte"] - TARGET_DTE),
                abs(long_leg["strike"] - spot),
                debit_dollars,
                expiration,
                long_leg["contract_symbol"],
                short_leg["contract_symbol"],
            )
            choices.append((key, expression))
    return sorted(choices, key=lambda item: item[0])


def build_publication(
    phase3: dict[str, Any],
    *,
    now: datetime,
    manifest: dict[str, Any] | None = None,
    option_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_phase3(phase3)
    early = _upstream_without_candidate(phase3, now)
    if early is not None:
        return early
    if manifest is None:
        raise ValueError(
            "candidate_published Phase 3 input requires the locked manifest"
        )
    candidate, slot = _validate_candidate_identity(manifest, phase3)
    expected_time = _expression_time(candidate)
    if now != expected_time:
        raise ValueError(
            "options expression must occur at prediction_ts plus one minute"
        )
    if now >= parse_timestamp(slot["entry_ts"], "entry_ts"):
        raise ValueError("options expression must occur before locked entry_ts")
    if candidate["decision"] == "stand_down":
        if option_snapshot is not None:
            raise ValueError("stand_down must not consume an option snapshot")
        return {
            **_base(phase3, now),
            "state": "abstained",
            "evaluator_accounting": "stand_down",
            "reason": "Phase 3 candidate stood down",
            "expression": None,
        }
    if option_snapshot is None:
        raise ValueError("directional candidate requires an exact option snapshot")
    spot, contracts = _validate_snapshot_identity(
        manifest, candidate, slot, option_snapshot, now
    )
    risk_cap = float(candidate["risk_cap_dollars"])
    choices = _spread_candidates(
        contracts,
        decision=candidate["decision"],
        session=date.fromisoformat(slot["session_date"]),
        captured=now,
        spot=spot,
        risk_cap=risk_cap,
    )
    snapshot_hash = _artifact_sha256(option_snapshot)
    if not choices:
        return {
            **_base(phase3, now),
            "state": "no_valid_expression",
            "evaluator_accounting": "zero_utility_rejection",
            "reason": "no adjacent liquid debit spread fit the locked risk cap",
            "option_snapshot_sha256": snapshot_hash,
            "expression": None,
        }
    expression = {
        "schema": EXPRESSION_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": candidate["slot_id"],
        "symbol": candidate["symbol"],
        "decision": candidate["decision"],
        "candidate_sha256": payload_sha256(candidate),
        "phase3_publication_sha256": _artifact_sha256(phase3),
        "option_snapshot_sha256": snapshot_hash,
        "expression_at": now.isoformat(),
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "variant_index": 1,
        "variant_count": 1,
        "risk_cap_dollars": risk_cap,
        **choices[0][1],
        "source_refs": [
            f"phase3-sha256://{_artifact_sha256(phase3)}",
            f"option-snapshot-sha256://{snapshot_hash}",
            str((option_snapshot.get("source") or {}).get("source_ref")),
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }
    expression["expression_id"] = payload_sha256(expression)
    return {
        **_base(phase3, now),
        "state": "expression_published",
        "evaluator_accounting": "candidate_pending_skeptic",
        "reason": "fixed adjacent-strike debit-spread rule selected one expression",
        "option_snapshot_sha256": snapshot_hash,
        "expression": expression,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-publication", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--option-snapshot", type=Path)
    parser.add_argument("--now")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit(
            "Options Expression Agent is artifact-only; --no-network is required"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only artifact: {args.output}")
    phase3 = json.loads(args.phase3_publication.read_text(encoding="utf-8"))
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    )
    snapshot = (
        json.loads(args.option_snapshot.read_text(encoding="utf-8"))
        if args.option_snapshot
        else None
    )
    if args.now:
        now = parse_timestamp(args.now, "now")
    elif phase3.get("state") == "candidate_published":
        raise SystemExit("directional/stand_down Phase 3 input requires explicit --now")
    else:
        now = parse_timestamp(phase3.get("published_at"), "published_at")
    artifact = build_publication(
        phase3, now=now, manifest=manifest, option_snapshot=snapshot
    )
    args.output.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    print(json.dumps({"state": artifact["state"], "slot_id": artifact["slot_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
