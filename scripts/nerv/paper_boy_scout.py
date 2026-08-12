"""NERV symbol-level scout for proposed paper-only Alpha Swarm lanes.

The scout answers one narrow question: which underlyings have enough observable
options liquidity to deserve a separately frozen Paper Boy experiment? It does
not choose a direction, contract, DTE, winner, or execution action.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

SCHEMA = "sharpedge.nerv_paper_boy_scout.v1"
RESEARCH_WARNING = (
    "Research-only nomination. A fresh provider snapshot and a separately locked "
    "paper-agent manifest are required before any candidate publication."
)


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def existing_symbol_lanes(catalyst_universe: dict[str, Any] | None) -> dict[str, str]:
    """Return symbols already active or queued in the catalyst-paper universe."""
    if not catalyst_universe:
        return {}
    lanes: dict[str, str] = {}
    for lane in catalyst_universe.get("lanes") or []:
        symbol = str(lane.get("symbol") or "").strip().upper()
        status = str(lane.get("status") or "unknown").strip()
        if symbol:
            lanes[symbol] = status
    return lanes


def build_scout_payload(
    board: dict[str, Any],
    *,
    catalyst_universe: dict[str, Any] | None = None,
    as_of: datetime | None = None,
    max_snapshot_age_minutes: float = 45.0,
    max_contract_age_seconds: int = 20 * 60,
    min_nerv_score: float = 65.0,
    min_usable_contracts: int = 3,
    limit: int = 5,
) -> dict[str, Any]:
    """Build a deterministic, symbol-level Paper Boy nomination report."""
    if board.get("schema") != "sharpedge.nerv_liquidity_board.v1":
        raise ValueError("unsupported NERV liquidity-board schema")

    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    summary = board.get("summary") or {}
    fetch_raw = str(summary.get("fetch_timestamp") or "").strip()
    fetch_at = parse_timestamp(fetch_raw) if fetch_raw else None
    snapshot_age_minutes = (
        max((now - fetch_at).total_seconds() / 60.0, 0.0) if fetch_at else None
    )
    snapshot_fresh = bool(
        snapshot_age_minutes is not None
        and snapshot_age_minutes <= max_snapshot_age_minutes
    )
    existing = existing_symbol_lanes(catalyst_universe)

    requested = {
        str(symbol).strip().upper()
        for symbol in summary.get("requested_symbols") or []
        if str(symbol).strip()
    }
    grouped: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in requested}
    for contract in board.get("contracts") or []:
        symbol = str(contract.get("underlying") or "").strip().upper()
        if symbol:
            grouped.setdefault(symbol, []).append(contract)

    rows = [
        _symbol_row(
            symbol,
            contracts,
            existing_status=existing.get(symbol),
            snapshot_fresh=snapshot_fresh,
            max_contract_age_seconds=max_contract_age_seconds,
            min_nerv_score=min_nerv_score,
            min_usable_contracts=min_usable_contracts,
        )
        for symbol, contracts in sorted(grouped.items())
    ]

    nominated = [row for row in rows if row["state"] == "nominated"]
    nominated.sort(
        key=lambda row: (
            row["top_contract_score_mean"],
            row["usable_contract_count"],
            row["best_nerv_score"],
            row["symbol"],
        ),
        reverse=True,
    )
    selected = nominated if limit <= 0 else nominated[:limit]
    selected_symbols = {row["symbol"] for row in selected}
    rank = {row["symbol"]: index for index, row in enumerate(selected, start=1)}
    for row in rows:
        if row["symbol"] in rank:
            row["nomination_rank"] = rank[row["symbol"]]
        elif row["state"] == "nominated":
            row["state"] = "below_nomination_cutoff"
            row["reasons"].append("outside configured nomination limit")

    return {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "source": {
            "schema": board.get("schema"),
            "provider": summary.get("source"),
            "data_mode": summary.get("data_mode"),
            "fetch_timestamp": fetch_raw or None,
            "snapshot_age_minutes": (
                round(snapshot_age_minutes, 3)
                if snapshot_age_minutes is not None
                else None
            ),
            "snapshot_fresh": snapshot_fresh,
            "errors": list(summary.get("errors") or []),
        },
        "criteria": {
            "max_snapshot_age_minutes": max_snapshot_age_minutes,
            "max_contract_age_seconds": max_contract_age_seconds,
            "min_nerv_score": min_nerv_score,
            "min_usable_contracts": min_usable_contracts,
            "nomination_limit": limit,
        },
        "existing_lanes": existing,
        "nominations": [row for row in rows if row["symbol"] in selected_symbols],
        "rows": rows,
        "summary": {
            "requested_symbol_count": len(rows),
            "nominated_symbol_count": len(selected),
            "nominated_symbols": [row["symbol"] for row in selected],
            "existing_lane_count": len(existing),
            "snapshot_fresh": snapshot_fresh,
        },
        "governance": {
            "role": "universe_nomination_only",
            "paper_only": True,
            "authoritative": False,
            "execution_permitted": False,
            "broker_access_allowed": False,
            "directional_output_allowed": False,
            "contract_selection_allowed": False,
            "dte_selection_allowed": False,
            "winner_selection_allowed": False,
            "aggregate_score_computed": False,
            "manifest_required_before_evidence": True,
            "warning": RESEARCH_WARNING,
        },
    }


def _symbol_row(
    symbol: str,
    contracts: list[dict[str, Any]],
    *,
    existing_status: str | None,
    snapshot_fresh: bool,
    max_contract_age_seconds: int,
    min_nerv_score: float,
    min_usable_contracts: int,
) -> dict[str, Any]:
    usable = [
        contract
        for contract in contracts
        if _contract_is_usable(contract, max_contract_age_seconds)
    ]
    top = sorted(
        usable,
        key=lambda contract: float(contract.get("nerv_score") or 0.0),
        reverse=True,
    )[:5]
    scores = [float(contract.get("nerv_score") or 0.0) for contract in top]
    best_score = max(scores, default=0.0)
    top_mean = mean(scores) if scores else 0.0
    reasons: list[str] = []

    if existing_status:
        state = "existing_or_queued_lane"
        reasons.append(f"already represented in catalyst universe: {existing_status}")
    elif not snapshot_fresh:
        state = "source_stale"
        reasons.append("NERV source snapshot exceeds the freshness limit")
    elif len(usable) < min_usable_contracts:
        state = "insufficient_usable_contracts"
        reasons.append(
            f"usable contracts {len(usable)} below minimum {min_usable_contracts}"
        )
    elif best_score < min_nerv_score:
        state = "below_quality_floor"
        reasons.append(
            f"best NERV score {best_score:.2f} below minimum {min_nerv_score:.2f}"
        )
    else:
        state = "nominated"
        reasons.append(
            "fresh multi-contract options liquidity merits paper-lane review"
        )

    return {
        "symbol": symbol,
        "state": state,
        "nomination_rank": None,
        "contract_count": len(contracts),
        "usable_contract_count": len(usable),
        "best_nerv_score": round(best_score, 2),
        "top_contract_score_mean": round(top_mean, 2),
        "usable_call_count": sum(row.get("option_type") == "call" for row in usable),
        "usable_put_count": sum(row.get("option_type") == "put" for row in usable),
        "expiration_count": len(
            {row.get("expiration") for row in usable if row.get("expiration")}
        ),
        "best_contract_observed": top[0].get("contract_symbol") if top else None,
        "existing_lane_status": existing_status,
        "reasons": reasons,
    }


def _contract_is_usable(contract: dict[str, Any], max_age_seconds: int) -> bool:
    flags = str(contract.get("rejection_flags") or "").strip()
    priority = str(contract.get("manual_validation_priority") or "").strip().lower()
    try:
        age = int(float(contract.get("quote_age_seconds")))
    except (TypeError, ValueError):
        return False
    return (
        not flags and priority in {"high", "medium", "low"} and age <= max_age_seconds
    )


def write_scout_artifacts(
    payload: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "latest.json"
    markdown_path = out / "latest.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NERV → Paper Boy Scout",
        "",
        f"**Nominations: {', '.join(payload['summary']['nominated_symbols']) or 'none'}**",
        "",
        "> Research-only universe nomination. No direction, contract, DTE, score, or execution authority.",
        "",
        f"- Source fresh: **{payload['source']['snapshot_fresh']}**",
        f"- Source age: `{payload['source']['snapshot_age_minutes']}` minutes",
        f"- Provider: `{payload['source']['provider']}`",
        "",
        "| Rank | Symbol | State | Usable | Best NERV | Top-5 mean | Why |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(
        payload["rows"],
        key=lambda item: (
            item["nomination_rank"] is None,
            item["nomination_rank"] or 999,
            item["symbol"],
        ),
    ):
        lines.append(
            "| {rank} | {symbol} | {state} | {usable} | {best:.2f} | {mean:.2f} | {why} |".format(
                rank=row["nomination_rank"] or "—",
                symbol=row["symbol"],
                state=row["state"],
                usable=row["usable_contract_count"],
                best=row["best_nerv_score"],
                mean=row["top_contract_score_mean"],
                why="; ".join(row["reasons"]),
            )
        )
    lines.extend(
        [
            "",
            "## Promotion boundary",
            "",
            "A nominated symbol is only a proposal. Promotion requires operator review, a new immutable manifest locked before evidence acquisition, and a separate paper-only worker.",
            "",
        ]
    )
    return "\n".join(lines)
