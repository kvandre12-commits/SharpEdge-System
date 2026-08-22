from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import confluence_zones as cz


def _signal(**over):
    base = {
        "spot": 765.72, "gamma_regime": "positive", "gamma_data_quality": "ok",
        "atm_straddle_mid": 3.4, "put_wall": 765.0, "pin": 765.0, "max_pain": 768.0,
        "call_wall": 772.0, "ema9": 766.0, "ema20": 766.11,
        "line_authority": {"lines": [{"name": "ORL", "price": 764.17, "score": 78, "event": "reclaimed"}]},
        "fair_value_gaps": {"open_gaps": []}, "level_states": {},
    }
    base.update(over)
    return base


def _overlay(**over):
    ov = {
        "schema": "sharpedge.confluence_zone_adjustments.v1",
        "authority": "diagnostic_shadow_overlay",
        "enabled": True,
        "multiplier_bounds": [0.7, 1.3],
        "adjustments": {"wall": {"multiplier": 1.3}, "pin": {"multiplier": 1.3}},
    }
    ov.update(over)
    return ov


def _points():
    return [
        {"name": "PUT_WALL", "kind": "wall", "price": 765.0, "weight": 1.0, "authority_multiplier": 1.0},
        {"name": "ORL", "kind": "reference", "price": 764.17, "weight": 0.6, "authority_multiplier": 0.78},
    ]


# --- pure overlay apply -----------------------------------------------------

def test_apply_overlay_scales_matching_kind_and_records(monkeypatch):
    monkeypatch.setenv(cz.REALTIME_ADJUST_ENV, "1")
    pts, applied = cz._apply_weight_overlay(_points(), _overlay())
    wall = next(p for p in pts if p["kind"] == "wall")
    assert wall["weight"] == 1.3  # 1.0 * 1.3
    assert next(p for p in pts if p["kind"] == "reference")["weight"] == 0.6  # untouched kind
    assert {a["kind"] for a in applied} == {"wall"}


def test_apply_overlay_clamps_to_bounds(monkeypatch):
    monkeypatch.setenv(cz.REALTIME_ADJUST_ENV, "1")
    pts, _ = cz._apply_weight_overlay(_points(), _overlay(adjustments={"wall": {"multiplier": 9.0}}))
    assert next(p for p in pts if p["kind"] == "wall")["weight"] == 1.3  # clamped 9.0 -> 1.3


def test_apply_overlay_rejects_foreign_schema_or_authority(monkeypatch):
    monkeypatch.setenv(cz.REALTIME_ADJUST_ENV, "1")
    _, a1 = cz._apply_weight_overlay(_points(), _overlay(schema="sharpedge.spine_adjustments.v1"))
    _, a2 = cz._apply_weight_overlay(_points(), _overlay(authority="something_else"))
    _, a3 = cz._apply_weight_overlay(_points(), _overlay(enabled=False))
    assert a1 == [] and a2 == [] and a3 == []


def test_apply_overlay_noop_when_env_disabled(monkeypatch):
    monkeypatch.delenv(cz.REALTIME_ADJUST_ENV, raising=False)
    pts, applied = cz._apply_weight_overlay(_points(), _overlay())
    assert applied == [] and next(p for p in pts if p["kind"] == "wall")["weight"] == 1.0


# --- end-to-end via build_confluence_zones ----------------------------------

def test_build_default_off_is_inert(monkeypatch):
    monkeypatch.delenv(cz.REALTIME_ADJUST_ENV, raising=False)
    out = cz.build_confluence_zones(_signal())
    assert out["realtime_adjustments"]["enabled"] is False
    assert out["realtime_adjustments"]["applied"] == []


def test_build_on_scales_conviction(monkeypatch):
    # negative gamma keeps the support cluster below the 100 cap so the
    # multiplier's effect on conviction is observable.
    sig = _signal(gamma_regime="negative")
    off = cz.build_confluence_zones(sig)
    monkeypatch.setenv(cz.REALTIME_ADJUST_ENV, "1")
    monkeypatch.setattr(cz, "_load_weight_overlay", lambda: _overlay())
    on = cz.build_confluence_zones(sig)
    assert on["realtime_adjustments"]["enabled"] is True
    assert on["realtime_adjustments"]["applied"], "expected wall/pin scaled"
    # the wall+pin support cluster should score higher with boosted wall/pin weight
    off_sup = next(z["conviction"] for z in off["zones"] if z["side"] == "support")
    on_sup = next(z["conviction"] for z in on["zones"] if z["side"] == "support")
    assert on_sup > off_sup


def test_load_overlay_empty_when_disabled(monkeypatch):
    monkeypatch.delenv(cz.REALTIME_ADJUST_ENV, raising=False)
    assert cz._load_weight_overlay() == {}
