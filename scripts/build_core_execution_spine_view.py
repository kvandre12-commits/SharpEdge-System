from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COCKPIT_DIR = ROOT / "cockpit"

sys.path.insert(0, str(COCKPIT_DIR))

from execution_vector_weights import DEFAULT_BASE_WEIGHTS  # noqa: E402

CORE_SPINE = [
    "structure_score",
    "acceptance_score",
    "trend_score",
    "location_score",
    "volume_score",
    "time_of_day_score",
    "dealer_gamma_score",
]

SECONDARY_CONFIRMATIONS = [
    "trap_score",
    "rejection_score",
]

CONTEXT_GOVERNORS = [
    "opening_auction_score",
    "exhaustion_score",
    "balance_context_score",
    "volatility_score",
    "compression_score",
]

SUSPECT_DRIFT = [
    "pressure_score",
    "regime_score",
]

LABEL_OVERRIDES = {
    "acceptance_score": "Auction Acceptance",
}

DECISION_LADDER = [
    {
        "question": "Can I trade?",
        "layer": "Trade Gate",
        "purpose": "permission / eligibility",
    },
    {
        "question": "Should I trade?",
        "layer": "Core Execution Spine",
        "purpose": "primary execution authority",
    },
    {
        "question": "Am I being fooled?",
        "layer": "Drift Voices",
        "purpose": "scheme-drift and overdescription checks",
    },
    {
        "question": "Is this setup confirmed?",
        "layer": "Trap / Rejection / Candle",
        "purpose": "confirmation and local response evidence",
    },
    {
        "question": "How should I manage risk?",
        "layer": "Governors",
        "purpose": "context, caution, and risk conditioning",
    },
]


def _label(name: str) -> str:
    return LABEL_OVERRIDES.get(
        name, name.replace("_score", "").replace("_", " ").title()
    )


def _score_row(name: str, permission: dict[str, Any]) -> dict[str, Any] | None:
    item = (permission.get("scores") or {}).get(name)
    if not item:
        return None
    return {
        "name": name,
        "label": _label(name),
        "score": int(item.get("score", 0)),
        "reason": str(item.get("reason", "")),
        "weight": float(DEFAULT_BASE_WEIGHTS.get(name, 0.0)),
    }


def _rows(names: list[str], permission: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        row = _score_row(name, permission)
        if row is not None:
            rows.append(row)
    return rows


def _normalized_weighted_score(rows: list[dict[str, Any]]) -> float:
    total_weight = sum(float(row.get("weight", 0.0)) for row in rows)
    if total_weight <= 0:
        return 0.0
    weighted = sum(float(row["score"]) * float(row["weight"]) for row in rows)
    return weighted / total_weight


def _sorted_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def build_view(signal: dict[str, Any]) -> dict[str, Any]:
    permission = signal.get("trade_permission") or {}
    core = _rows(CORE_SPINE, permission)
    secondary = _rows(SECONDARY_CONFIRMATIONS, permission)
    context = _rows(CONTEXT_GOVERNORS, permission)
    drift = _rows(SUSPECT_DRIFT, permission)
    core_ranked = _sorted_scores(core)

    return {
        "schema": "sharpedge.core_execution_spine_view.v1",
        "ts": signal.get("ts"),
        "symbol": signal.get("symbol"),
        "spot": signal.get("spot"),
        "engine": {
            "trade_permission_score": permission.get("trade_permission_score"),
            "execution_permission_score": permission.get("execution_permission_score"),
            "trade_gate": permission.get("trade_gate"),
            "bias": permission.get("bias"),
            "bias_strength": permission.get("bias_strength"),
        },
        "core_spine": {
            "normalized_weighted_score": round(_normalized_weighted_score(core), 2),
            "features": core,
            "best": core_ranked[:3],
            "worst": list(reversed(core_ranked[-3:])),
        },
        "secondary_confirmations": secondary,
        "context_governors": context,
        "suspect_drift_voices": drift,
        "decision_ladder": DECISION_LADDER,
        "audit_notes": [
            "This view restores hierarchy by treating the core spine as the authority layer.",
            "Acceptance is labeled Auction Acceptance here to separate level-acceptance evidence from pure market structure.",
            "Trap and Rejection stay visible as confirmations, not core spine inputs.",
            "Pressure and Regime are separated as suspect drift voices rather than spine authorities.",
            "This spine score intentionally does not inherit the engine's acceptance max(acceptance, rejection, trap) score bucket behavior.",
            "Current spine weights remain fixed for readability; future doctrine could make them regime-aware without changing this audit surface first.",
        ],
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_none_\n"
    lines = [
        "| Feature | Score | Base Weight | Reason |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['score']} | {row['weight']:.2f} | {row['reason']} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_bullets(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- none\n"
    return (
        "\n".join(
            f"- **{row['label']} {row['score']}** — {row['reason']}" for row in rows
        )
        + "\n"
    )


def render_markdown(view: dict[str, Any]) -> str:
    engine = view["engine"]
    core = view["core_spine"]
    lines = [
        "# Core Execution Spine View",
        "",
        f"- ts: `{view.get('ts')}`",
        f"- symbol: `{view.get('symbol')}`",
        f"- spot: `{view.get('spot')}`",
        f"- engine trade gate: **{engine.get('trade_gate')}**",
        f"- engine permission: **{engine.get('trade_permission_score')}**",
        f"- engine bias: **{engine.get('bias')}** ({engine.get('bias_strength')})",
        f"- core spine normalized weighted score: **{core.get('normalized_weighted_score')}**",
        "",
        "## Core spine authority layer",
        "",
        _markdown_table(core["features"]),
        "## Core spine best",
        "",
        _markdown_bullets(core["best"]),
        "## Core spine weakest",
        "",
        _markdown_bullets(core["worst"]),
        "## Secondary confirmations",
        "",
        _markdown_table(view["secondary_confirmations"]),
        "## Context / governors",
        "",
        _markdown_table(view["context_governors"]),
        "## Suspect drift voices",
        "",
        _markdown_table(view["suspect_drift_voices"]),
        "## Decision ladder",
        "",
        *(
            f"- **{step['question']}** → **{step['layer']}** — {step['purpose']}"
            for step in view["decision_ladder"]
        ),
        "",
        "## Audit notes",
        "",
        *(f"- {note}" for note in view["audit_notes"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    signal_path = ROOT / "outputs" / "signal.json"
    output_json = ROOT / "outputs" / "core_execution_spine_view.json"
    output_md = ROOT / "outputs" / "core_execution_spine_view.md"

    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    view = build_view(signal)
    output_json.write_text(json.dumps(view, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(view), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
