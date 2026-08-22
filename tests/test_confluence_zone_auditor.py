from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import confluence_zone_auditor as az


def _snap(ts, spot, zones):
    return {"ts": ts, "spot": spot, "gamma_regime": "negative", "zones": zones}


def _zone(side="support", lo=764.0, hi=765.0, kinds=("wall", "pin")):
    return {"zone_id": "S1", "side": side, "zone_lo": lo, "zone_hi": hi,
            "conviction_band": "medium", "kinds": list(kinds)}


# --- snapshot reduction -----------------------------------------------------

def test_snapshot_from_signal_extracts_zone_kinds():
    signal = {"spot": 765.72, "gamma_regime": "negative", "confluence_zones": {
        "gamma_regime": "negative",
        "zones": [{"zone_id": "R1", "side": "resistance", "zone_lo": 766.0, "zone_hi": 766.5,
                   "conviction_band": "high",
                   "contributing_factors": [{"kind": "vwap"}, {"kind": "ema"}, {"kind": "ema"}]}]}}
    snap = az.snapshot_from_signal(signal, "2026-08-22T15:00:00Z")
    assert snap["spot"] == 765.72
    assert snap["zones"][0]["kinds"] == ["ema", "vwap"]  # deduped + sorted


def test_snapshot_from_signal_skips_pre_feature_or_unpriced():
    # no confluence_zones field at all (pre-feature) -> skipped
    assert az.snapshot_from_signal({"spot": 1.0}, "t") is None
    # unpriced -> skipped
    assert az.snapshot_from_signal({"confluence_zones": {"zones": []}}, "t") is None
    # field present but empty zones + priced -> kept (feeds forward spot path)
    kept = az.snapshot_from_signal({"spot": 1.0, "confluence_zones": {"zones": []}}, "t")
    assert kept is not None and kept["zones"] == []


# --- grading ----------------------------------------------------------------

def _grade(zone, spot_path):
    # origin carries the zone; later snaps supply the forward spot path
    snaps = [_snap("2026-08-22T15:00:00Z", spot_path[0], [zone])]
    for i, sp in enumerate(spot_path[1:], start=1):
        snaps.append(_snap(f"2026-08-22T15:{i:02d}:00Z", sp, []))
    out = az.grade_zone_outcomes(snaps, horizon_seconds=3600)
    return out[0]["result"]


def test_grade_respected_support_bounce():
    # dips into [764,765], then reclaims above 765 + buffer
    assert _grade(_zone(), [765.5, 764.5, 765.6]) == "RESPECTED"


def test_grade_broken_support():
    # dips into zone, then accepts below 764 - buffer
    assert _grade(_zone(), [765.5, 764.5, 763.4]) == "BROKEN"


def test_grade_untested_when_never_touched():
    assert _grade(_zone(), [766.0, 766.5, 767.0]) == "UNTESTED"


def test_grade_resistance_respected_on_rejection():
    z = _zone(side="resistance", lo=766.0, hi=766.5)
    # pushes into [766,766.5] then fails below 766 - buffer
    assert _grade(z, [765.5, 766.2, 765.4]) == "RESPECTED"


# --- attribution + de-bias --------------------------------------------------

def test_summary_debiases_ubiquitous_kind():
    # 'vwap' in every zone -> rate == baseline -> multiplier ~1.0
    # 'wall' only in respected zones -> rate 1.0 -> boosted
    outcomes = []
    for _ in range(3):
        outcomes.append({"result": "RESPECTED", "kinds": ["vwap", "wall"]})
    for _ in range(3):
        outcomes.append({"result": "BROKEN", "kinds": ["vwap"]})
    summary = az.summarize_factor_edges(outcomes, min_samples=2, gain=1.5)
    adj = summary["adjustments"]
    assert summary["baseline_respected_rate"] == 0.5
    assert adj["vwap"]["multiplier"] == 1.0  # ubiquitous, at baseline
    assert adj["wall"]["multiplier"] > 1.0   # only in winners


def test_summary_min_samples_gate_holds_at_one():
    outcomes = [{"result": "RESPECTED", "kinds": ["wall"]}]
    adj = az.summarize_factor_edges(outcomes, min_samples=5)["adjustments"]
    assert adj["wall"]["multiplier"] == 1.0
    assert "insufficient" in adj["wall"]["action"]


def test_summary_multiplier_clamped_to_bounds():
    # wall always wins; a separate 'noise' kind loses -> baseline < 1.0 so wall lifts
    outcomes = [{"result": "RESPECTED", "kinds": ["wall"]} for _ in range(20)]
    outcomes += [{"result": "BROKEN", "kinds": ["noise"]} for _ in range(20)]
    adj = az.summarize_factor_edges(outcomes, min_samples=2, gain=99, bounds=(0.7, 1.3))["adjustments"]
    assert adj["wall"]["multiplier"] == 1.3  # clamped high


# --- overlay + end-to-end ---------------------------------------------------

def test_build_overlay_shape_and_enabled_flag():
    outcomes = [{"result": "RESPECTED", "kinds": ["wall"]} for _ in range(20)]
    outcomes += [{"result": "BROKEN", "kinds": ["noise"]} for _ in range(20)]
    summary = az.summarize_factor_edges(outcomes, min_samples=2)
    overlay = az.build_weight_overlay(summary, min_samples=2)
    assert overlay["schema"] == az.ADJUSTMENT_SCHEMA
    assert overlay["authority"] == az.SHADOW_AUTHORITY
    assert overlay["enabled"] is True
    assert "wall" in overlay["adjustments"]


def test_run_once_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    # two snapshots: origin with a support zone, then a reclaim spot
    for i, (ts_dir, spot, zones) in enumerate([
        ("20260822T150000Z", 765.5, [{"zone_id": "S1", "side": "support", "zone_lo": 764.0,
          "zone_hi": 765.0, "conviction_band": "medium",
          "contributing_factors": [{"kind": "wall"}, {"kind": "pin"}]}]),
        ("20260822T150500Z", 764.5, []),
        ("20260822T151000Z", 765.8, []),
    ]):
        d = cache / ts_dir / "outputs"
        d.mkdir(parents=True)
        sig = {"spot": spot, "gamma_regime": "negative", "confluence_zones": {"zones": zones, "gamma_regime": "negative"}}
        (d / "signal.json").write_text(json.dumps(sig))
    overlay = az.run_once(
        cache_dir=cache,
        signal_path=tmp_path / "none.json",
        ledger_path=tmp_path / "led.jsonl",
        output_dir=tmp_path / "out",
        adjustment_path=tmp_path / "adj.json",
        min_samples=1,
    )
    assert overlay["schema"] == az.ADJUSTMENT_SCHEMA
    assert (tmp_path / "adj.json").exists()
    assert (tmp_path / "out" / "latest.txt").exists()
    # wall+pin were in a RESPECTED zone -> present in adjustments
    assert "wall" in overlay["adjustments"] and "pin" in overlay["adjustments"]
