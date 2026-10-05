"""Confluence bounce/rejection zones for `sharpedge.signal.v1`.

Stacks the level/indicator signals SharpEdge already computes into CLUSTERED
price zones, scores each zone's confluence, gates by gamma regime, sizes with the
market-priced expected move, and emits ranked bounce (support) / rejection
(resistance) zones — each with a trigger and an invalidation.

Advisory only. `weighted_in_permission` is False; this never feeds `decide()` or
the permission spine. Pure: `build_confluence_zones(signal_dict) -> dict`.

Two factor sources, deliberately (line_authority scores only rails, never the
walls/pin/EMAs/gaps):
  * scored  — `signal["line_authority"]["lines"]`: rails carry their 0-100
              authority as a multiplier + acceptance event. Never recomputed.
  * static  — top-level signal fields (walls, pin, max_pain, ema9/20, open FVG
              midpoints): fixed authority weight, no acceptance event.
`level_states` is consulted for trigger/invalidation freshness ONLY — never as a
scoring factor — so acceptance is not double-counted with line_authority.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "sharpedge.confluence_zones.v1"

# Opt-in self-audit weight overlay (mirrors the spine's realtime-adjust seam).
# Default OFF: weights stay code-default unless SHARPEDGE_CONFLUENCE_REALTIME_ADJUST=1.
REALTIME_ADJUST_ENV = "SHARPEDGE_CONFLUENCE_REALTIME_ADJUST"
REALTIME_ADJUST_PATH_ENV = "SHARPEDGE_CONFLUENCE_REALTIME_ADJUSTMENTS"
DEFAULT_CONFLUENCE_ADJUSTMENT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "confluence_zone_adjustments.json"
)
_ADJUSTMENT_SCHEMA = "sharpedge.confluence_zone_adjustments.v1"
_SHADOW_AUTHORITY = "diagnostic_shadow_overlay"
_DEFAULT_MULTIPLIER_BOUNDS = (0.7, 1.3)

# Static-factor authority weights (operator-tunable).
_STATIC_FACTORS: dict[str, tuple[str, float]] = {
    "put_wall": ("wall", 1.0),
    "call_wall": ("wall", 1.0),
    "pin": ("pin", 1.0),
    "max_pain": ("magnet", 0.7),
    "ema9": ("ema", 0.4),
    "ema20": ("ema", 0.4),
}

# Scored-line (line_authority) weights by rail name.
_LINE_WEIGHTS: dict[str, tuple[str, float]] = {
    "BALANCE_HIGH": ("balance_edge", 0.7),
    "BALANCE_LOW": ("balance_edge", 0.7),
    "BALANCE_MID": ("reference", 0.6),
    "DAY_MID": ("reference", 0.6),
    "VWAP": ("vwap", 0.6),
    "ORH": ("reference", 0.6),
    "ORL": ("reference", 0.6),
    "PDH": ("reference", 0.6),
    "PDL": ("reference", 0.6),
    "PDC": ("reference", 0.6),
}
_FVG_WEIGHT = ("fvg", 0.5)

# Normalization + gating constants.
_EXPECTED_MAX = 3.0            # raw score that maps to conviction 100
_PROXIMITY_BONUS_MAX = 12.0    # added for zones sitting right on spot
_WALL_PROXIMITY_PCT = 0.20     # a zone within this % of a wall is flagged pinned
_REGIME_MULTIPLIERS = {
    "boost": 1.15,
    "neutral": 1.0,
    "penalty": 0.6,
    "trap_veto": 0.3,
}


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _confluence_overlay_enabled() -> bool:
    return str(os.getenv(REALTIME_ADJUST_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_weight_overlay() -> dict[str, Any]:
    """Load the self-audit weight overlay, or {} unless opted in."""
    if not _confluence_overlay_enabled():
        return {}
    path = Path(os.getenv(REALTIME_ADJUST_PATH_ENV) or DEFAULT_CONFLUENCE_ADJUSTMENT_PATH)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _apply_weight_overlay(
    points: list[dict[str, Any]], overlay: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scale static factor weights by a learned per-kind multiplier (opt-in, clamped).

    Gated on schema + authority + enabled + env flag so a spine overlay can never
    be consumed as confluence weights. Only the static ``weight`` is scaled;
    live acceptance (``authority_multiplier``) is never touched.
    """
    applied: list[dict[str, Any]] = []
    if (
        not overlay
        or overlay.get("schema") != _ADJUSTMENT_SCHEMA
        or overlay.get("authority") != _SHADOW_AUTHORITY
        or not overlay.get("enabled")
        or not _confluence_overlay_enabled()
    ):
        return points, applied
    adjustments = overlay.get("adjustments") or {}
    bounds = overlay.get("multiplier_bounds") or list(_DEFAULT_MULTIPLIER_BOUNDS)
    lo, hi = float(bounds[0]), float(bounds[1])
    for point in points:
        adj = adjustments.get(point["kind"])
        mult = (adj or {}).get("multiplier")
        if not isinstance(mult, (int, float)):
            continue
        clamped = max(lo, min(hi, float(mult)))
        point["weight"] = round(point["weight"] * clamped, 4)
        applied.append({"kind": point["kind"], "name": point["name"], "multiplier": clamped})
    return points, applied


def _collect_factor_points(signal: dict[str, Any]) -> list[dict[str, Any]]:
    """Gather scored + static factor points, deduped, deterministically sorted."""
    points: list[dict[str, Any]] = []
    seen_prices: set[float] = set()

    # Scored rails from line_authority (authority multiplier = score/100).
    la = signal.get("line_authority") or {}
    for line in la.get("lines") or []:
        name = str(line.get("name") or "")
        price = _num(line.get("price"))
        if price is None or name not in _LINE_WEIGHTS:
            continue
        kind, weight = _LINE_WEIGHTS[name]
        score = _num(line.get("score"))
        points.append({
            "name": name,
            "kind": kind,
            "price": price,
            "weight": weight,
            "authority_multiplier": round((score or 0.0) / 100.0, 4) if score is not None else 1.0,
            "event": line.get("event"),
            "source": "line_authority",
        })
        seen_prices.add(round(price, 4))

    # Static magnets/EMAs (fixed weight, no acceptance event). line_authority wins
    # on an exact price collision so acceptance is never double-counted.
    for field, (kind, weight) in _STATIC_FACTORS.items():
        price = _num(signal.get(field))
        if price is None or round(price, 4) in seen_prices:
            continue
        points.append({
            "name": field.upper(),
            "kind": kind,
            "price": price,
            "weight": weight,
            "authority_multiplier": 1.0,
            "event": None,
            "source": "signal",
        })

    # Open fair-value-gap midpoints (unfilled imbalances draw price).
    fvg = signal.get("fair_value_gaps") or {}
    for gap in fvg.get("open_gaps") or []:
        price = _num(gap.get("midpoint"))
        if price is None or round(price, 4) in seen_prices:
            continue
        kind, weight = _FVG_WEIGHT
        points.append({
            "name": f"FVG_{gap.get('direction', '')}".upper().rstrip("_"),
            "kind": kind,
            "price": price,
            "weight": weight,
            "authority_multiplier": 1.0,
            "event": gap.get("fill_state"),
            "source": "signal",
        })

    points.sort(key=lambda p: (p["price"], p["name"]))
    return points


def _expected_move(signal: dict[str, Any]) -> dict[str, Any]:
    """Market-priced expected move in dollars (straddle-first; IV is context)."""
    straddle = _num(signal.get("atm_straddle_mid"))
    atm_iv = _num(signal.get("atm_iv"))
    if straddle and straddle > 0:
        return {"straddle_mid": straddle, "atm_iv": atm_iv, "source": "straddle", "dollars": straddle}
    return {"straddle_mid": straddle, "atm_iv": atm_iv, "source": "none", "dollars": None}


def _cluster_tolerance(em_dollars: float | None, spot: float | None) -> tuple[float, str]:
    """Adaptive cluster tolerance: a fraction of the expected move, bounded."""
    spot = spot or 0.0
    floor = 0.0005 * spot
    cap = 0.004 * spot
    if em_dollars and em_dollars > 0:
        tol = max(floor, min(cap, 0.25 * em_dollars))
        return round(tol, 4), "0.25x_expected_move"
    return round(0.0015 * spot, 4), "fixed_0.15pct_fallback"


def _cluster(points: list[dict[str, Any]], tol: float) -> list[list[dict[str, Any]]]:
    """Single-pass agglomeration of price-sorted points within ``tol``."""
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for point in points:  # points arrive price-sorted
        if current and point["price"] - current[-1]["price"] > tol:
            clusters.append(current)
            current = []
        current.append(point)
    if current:
        clusters.append(current)
    return clusters


def _score_zone(cluster: list[dict[str, Any]], spot: float | None, em_dollars: float | None) -> dict[str, Any]:
    """Additive, explainable conviction: Σ(weight × authority) + proximity bonus."""
    raw = sum(p["weight"] * p["authority_multiplier"] for p in cluster)
    base = min(100.0, raw / _EXPECTED_MAX * 100.0)
    midpoint = sum(p["price"] for p in cluster) / len(cluster)
    proximity_bonus = 0.0
    if spot and em_dollars and em_dollars > 0:
        distance = abs(midpoint - spot)
        proximity_bonus = _PROXIMITY_BONUS_MAX * max(0.0, 1.0 - distance / (2.0 * em_dollars))
    return {
        "raw": round(raw, 4),
        "base": round(base, 2),
        "proximity_bonus": round(proximity_bonus, 2),
        "pre_regime": round(min(100.0, base + proximity_bonus), 2),
        "midpoint": round(midpoint, 2),
    }


def _side_stance(midpoint: float, spot: float) -> tuple[str, str]:
    """Geometry sets side/stance; regime only adjusts conviction, never the label."""
    if midpoint < spot:
        return "support", "bounce"
    return "resistance", "rejection"


def _level_state_for(cluster: list[dict[str, Any]], level_states: dict[str, Any]) -> dict[str, Any]:
    for point in cluster:
        state = level_states.get(point["name"])
        if state:
            return state
    return {}


def _apply_regime_gate(
    side: str,
    cluster: list[dict[str, Any]],
    regime: str,
    level_states: dict[str, Any],
) -> dict[str, Any]:
    """Positive gamma boosts fade-to-magnet; negative gamma traps naive dip-buys."""
    if regime == "positive":
        return {"applied": "boost", "multiplier": _REGIME_MULTIPLIERS["boost"],
                "reason": "positive gamma: fade toward magnet is reliable"}
    if regime == "negative":
        if side == "support":
            state = _level_state_for(cluster, level_states)
            reclaimed = str(state.get("event_state") or "") == "failed_break_reclaimed"
            window_open = bool(state.get("entry_window_open"))
            if reclaimed and window_open:
                return {"applied": "penalty", "multiplier": _REGIME_MULTIPLIERS["penalty"],
                        "reason": "negative gamma but a reclaim is confirmed (trap un-vetoed)"}
            return {"applied": "trap_veto", "multiplier": _REGIME_MULTIPLIERS["trap_veto"],
                    "reason": "negative gamma: dip-buy at support is a trap without a confirmed reclaim"}
        return {"applied": "penalty", "multiplier": _REGIME_MULTIPLIERS["penalty"],
                "reason": "negative gamma: rejection can fail as momentum runs"}
    return {"applied": "neutral", "multiplier": _REGIME_MULTIPLIERS["neutral"],
            "reason": "gamma regime unknown: no adjustment"}


def _near_wall(midpoint: float, signal: dict[str, Any]) -> bool:
    for field in ("put_wall", "call_wall"):
        wall = _num(signal.get(field))
        if wall and wall != 0 and abs(midpoint - wall) / wall * 100.0 <= _WALL_PROXIMITY_PCT:
            return True
    return False


def _band(conviction: float) -> str:
    if conviction >= 66:
        return "high"
    if conviction >= 40:
        return "medium"
    return "low"


def _trigger_invalidation(stance: str, zone_lo: float, zone_hi: float, regime: str) -> tuple[str, str]:
    if stance == "bounce":
        confirm = " (reclaim required in negative gamma)" if regime == "negative" else ""
        return (
            f"1m close reclaiming and holding above {zone_hi:.2f}{confirm}",
            f"1m close accepted below {zone_lo:.2f} (zone lost)",
        )
    return (
        f"1m close rejecting from and failing back below {zone_lo:.2f}",
        f"1m close accepted above {zone_hi:.2f} (zone reclaimed)",
    )


def build_confluence_zones(signal: dict[str, Any]) -> dict[str, Any]:
    """Assemble the ranked confluence-zone map. Fail-soft to empty zones."""
    spot = _num(signal.get("spot"))
    regime = str(signal.get("gamma_regime") or "unknown").lower()
    data_quality = str(signal.get("gamma_data_quality") or "ok").lower()
    em = _expected_move(signal)
    tol, tol_source = _cluster_tolerance(em["dollars"], spot)

    base = {
        "schema": SCHEMA,
        "advisory_role": "bounce_rejection_map_advisory",
        "weighted_in_permission": False,
        "spot": spot,
        "gamma_regime": regime,
        "data_quality": data_quality,
        "expected_move": em,
        "clustering": {"tolerance": tol, "tolerance_source": tol_source},
        "realtime_adjustments": {
            "enabled": _confluence_overlay_enabled(),
            "applied": [],
            "authority": _SHADOW_AUTHORITY,
        },
        "zones": [],
        "summary": {},
    }
    if spot is None or data_quality in {"missing", "expired"}:
        base["summary"] = {"reason": f"no zones: spot/{data_quality} data unavailable"}
        return base

    level_states = signal.get("level_states") or {}
    points = _collect_factor_points(signal)
    points, overlay_applied = _apply_weight_overlay(points, _load_weight_overlay())
    base["realtime_adjustments"]["applied"] = overlay_applied
    zones: list[dict[str, Any]] = []
    for idx, cluster in enumerate(_cluster(points, tol), start=1):
        score = _score_zone(cluster, spot, em["dollars"])
        midpoint = score["midpoint"]
        side, stance = _side_stance(midpoint, spot)
        gate = _apply_regime_gate(side, cluster, regime, level_states)
        conviction = round(min(100.0, score["pre_regime"] * gate["multiplier"]), 1)
        zone_lo = round(min(p["price"] for p in cluster), 2)
        zone_hi = round(max(p["price"] for p in cluster), 2)
        trigger, invalidation = _trigger_invalidation(stance, zone_lo, zone_hi, regime)
        zones.append({
            "zone_id": f"{'S' if side == 'support' else 'R'}{idx}",
            "side": side,
            "stance": stance,
            "zone_lo": zone_lo,
            "zone_hi": zone_hi,
            "midpoint": midpoint,
            "distance_pct": round((midpoint - spot) / spot * 100.0, 3) if spot else None,
            "conviction": conviction,
            "conviction_band": _band(conviction),
            "regime_gate": gate,
            "pinned_to_wall": _near_wall(midpoint, signal),
            "contributing_factors": cluster,
            "factor_count": len(cluster),
            "score_terms": score,
            "trigger": trigger,
            "invalidation": invalidation,
        })

    zones.sort(key=lambda z: (-z["conviction"], abs(z["distance_pct"] or 0.0)))
    for rank, zone in enumerate(zones, start=1):
        zone["rank"] = rank

    top_bounce = next((z["zone_id"] for z in zones if z["side"] == "support"), None)
    top_reject = next((z["zone_id"] for z in zones if z["side"] == "resistance"), None)
    base["zones"] = zones
    base["summary"] = {
        "top_bounce": top_bounce,
        "top_rejection": top_reject,
        "zone_count": len(zones),
        "reason": f"{len(zones)} confluence zones, {regime} gamma",
    }
    return base


__all__ = ["SCHEMA", "build_confluence_zones"]
