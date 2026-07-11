"""SharpEdge live cockpit for fast SPY reads from Yahoo 1m + CBOE options."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from ace_snapshot import write_ace_snapshot
from balance import build_balance_stack
from context_attachment import build_context_attachment
from decision_receipts import (
    append_decision_receipt,
    build_decision_receipt,
    build_permission_score_trend,
    load_recent_receipts,
)
from gamma import gamma_card, gamma_profile
from gate_workflows import primary_context_setup, primary_trade_setup
from level_state_engine import build_level_state_map
from live_chart_svg import chart_svg
from live_read_view import infer_target, render_live_read_html
from market_data_sources import (
    fetch_cboe_options_book,
    fetch_yahoo_intraday_session_rows,
    read_options_surface,
)
from monthly_context_chart import build_monthly_context_svg
from weekly_context_chart import build_weekly_context_svg
from range_posture import build_range_posture
from volume_profile import build_volume_profile
from vwap_posture import build_vwap_posture
from setups import (
    detect_exhaustion,
    detect_failed_breaks,
    detect_negative_gamma_continuation,
    detect_sticky_noise,
    detect_volatility_coil,
    read_volatility_structure,
    reference_levels,
)
from regime_refinement import annotate_market_behavior
from timeframe_agreement import build_timeframe_agreement
from transition_pressure import build_transition_pressure_packet
from runner_handoff_live import render_runner_handoff_live_html
from setup_event_lifecycle import annotate_setup_conviction, primary_setup_event
from trade_permission import score_trade_permission

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- data fetch -----------------------------
def fetch_intraday():
    rows, _ = fetch_yahoo_intraday_session_rows("SPY")
    return rows


def fetch_intraday_with_source():
    return fetch_yahoo_intraday_session_rows("SPY")


def fetch_options():
    spot, book, _ = fetch_cboe_options_book("SPY")
    return spot, book


def fetch_options_with_source():
    return fetch_cboe_options_book("SPY")


# ----------------------------- analytics -----------------------------
def read_price_action(rows):
    closes = [b[4] for b in rows]
    vols = [b[5] for b in rows]
    spot = closes[-1]
    day_open = closes[0]
    hi, lo = max(closes), min(closes)
    rng = (hi - lo) or 1e-9
    rng_pos = (spot - lo) / rng * 100  # 0=low, 100=high of day
    balance = build_balance_stack(rows)

    # VWAP (who controls the day)
    cum_pv = sum(b[4] * b[5] for b in rows)
    cum_v = sum(vols) or 1
    vwap = cum_pv / cum_v

    # short-term momentum: last 15 minutes
    look = min(15, len(closes) - 1)
    mom = (spot / closes[-1 - look] - 1) * 100 if look else 0.0

    volume_profile = build_volume_profile(rows)
    vol_mult = volume_profile["composite_mult"]

    return {
        "spot": spot,
        "day_open": day_open,
        "hi": hi,
        "lo": lo,
        **balance,
        "rng_pos": rng_pos,
        "day_chg": (spot / day_open - 1) * 100,
        "vwap": vwap,
        "vs_vwap": (spot - vwap) / vwap * 100,
        "mom15": mom,
        "vol_mult": vol_mult,
        "volume_profile": volume_profile,
    }


def read_options(spot, book):
    return read_options_surface(spot, book)


def synthesize(pa, op):
    """Plain-English, number-backed reads. Each line cites its data."""
    lines = []

    # 1. who controls the tape
    vwap_posture = build_vwap_posture(pa)
    range_posture = build_range_posture(pa, vwap_posture=vwap_posture)
    if vwap_posture["has_upside_control"]:
        lines.append(
            (
                "BULLS in control",
                "ok",
                f"price ${pa['spot']:.2f} is {pa['vs_vwap']:+.2f}% "
                f"ABOVE VWAP ${pa['vwap']:.2f}",
            )
        )
    elif vwap_posture["has_downside_control"]:
        lines.append(
            (
                "BEARS in control",
                "bad",
                f"price ${pa['spot']:.2f} is {pa['vs_vwap']:+.2f}% "
                f"BELOW VWAP ${pa['vwap']:.2f}",
            )
        )
    else:
        label = "hugging" if vwap_posture["state"] == "hugging_vwap" else "near"
        lines.append(
            (
                "BALANCED / chop",
                "warn",
                f"price {label} VWAP ${pa['vwap']:.2f} "
                f"({pa['vs_vwap']:+.2f}%) - no edge, wait",
            )
        )

    # 2. where in the day's range
    rp = pa["rng_pos"]
    if (
        bool(range_posture.get("is_pressing_edge"))
        and str(range_posture.get("side")) == "upside"
    ):
        lines.append(
            (
                "At day HIGHS",
                "warn",
                f"{rp:.0f}% of range | {pa['balance_state']} {pa['balance_reference']} balance at {pa['position_in_balance']:.2f} - breakout OR exhaustion zone",
            )
        )
    elif (
        bool(range_posture.get("is_pressing_edge"))
        and str(range_posture.get("side")) == "downside"
    ):
        lines.append(
            (
                "At day LOWS",
                "warn",
                f"{rp:.0f}% of range | {pa['balance_state']} {pa['balance_reference']} balance at {pa['position_in_balance']:.2f} - breakdown OR reclaim zone",
            )
        )
    else:
        lines.append(
            (
                "Mid-range",
                "info",
                f"{rp:.0f}% of day range | {pa['balance_state']} {pa['balance_reference']} balance at {pa['position_in_balance']:.2f} "
                f"(lo ${pa['balance_low']:.2f} / hi ${pa['balance_high']:.2f})",
            )
        )

    # 3. momentum real or fading
    if abs(pa["mom15"]) < 0.05:
        lines.append(
            ("Momentum FLAT", "info", f"{pa['mom15']:+.2f}% last 15m - no thrust")
        )
    elif pa["mom15"] > 0:
        lines.append(("Momentum UP", "ok", f"{pa['mom15']:+.2f}% last 15m"))
    else:
        lines.append(("Momentum DOWN", "bad", f"{pa['mom15']:+.2f}% last 15m"))

    # 4. volume confirming?
    vm = pa["vol_mult"]
    volume_profile = pa.get("volume_profile") or {}
    confirmation = volume_profile.get("confirmation")
    detail = volume_profile.get("reason") or f"{vm:.1f}x blended participation"
    if confirmation == "confirmed":
        lines.append(("Move volume CONFIRMS", "ok", detail))
    elif confirmation == "participating":
        lines.append(("Move volume participating", "info", detail))
    elif confirmation == "missing":
        lines.append(("Move volume MISSING", "warn", detail + " - fade/trap risk"))
    else:
        lines.append(("Move volume mixed", "warn", detail))

    # 5. options walls (magnets / levels)
    cw, pw = op["call_wall"], op["put_wall"]
    if cw is not None and pw is not None:
        lines.append(
            (
                "Options box",
                "info",
                f"put wall ${pw:g} (support) <-> call wall ${cw:g} "
                f"(resistance) | exp {op['exp']}",
            )
        )
    lines.append(
        (
            "Sentiment",
            "info",
            f"P/C OI {op['pcr']:.2f} | ATM IV {op['atm_iv'] * 100:.1f}%",
        )
    )

    return lines


# ----------------------------- rendering -----------------------------
def read_microstructure(rows, lookback=8):
    """OHLC-only microstructure of the session-so-far candle + a Donchian channel.

    rows = (minute, open, high, low, close, volume). All pure OHLC logic:
      - bar anatomy: lower/upper wick + body as % of the day range (wick =
        absorption/rejection; lower_wick is our strongest model feature).
      - Donchian channel over the last `lookback` bars: where price sits in the
        channel (0=floor,100=ceiling), channel width %, and channel slope.
    """
    if not rows:
        return {}
    o = rows[0][1]
    hi = max(r[2] for r in rows)
    lo = min(r[3] for r in rows)
    c = rows[-1][4]
    rng = max(hi - lo, 1e-9)
    lower_wick = (min(o, c) - lo) / rng * 100
    upper_wick = (hi - max(o, c)) / rng * 100
    body = abs(c - o) / rng * 100

    win = rows[-lookback:] if len(rows) >= lookback else rows
    ch_hi = max(r[2] for r in win)
    ch_lo = min(r[3] for r in win)
    ch_w = max(ch_hi - ch_lo, 1e-9)
    ch_pos = (c - ch_lo) / ch_w * 100
    ch_width_pct = ch_w / c * 100
    # channel slope: midline now vs midline `lookback` bars earlier
    prev = rows[-2 * lookback : -lookback] if len(rows) >= 2 * lookback else rows[:1]
    prev_mid = (max(r[2] for r in prev) + min(r[3] for r in prev)) / 2
    cur_mid = (ch_hi + ch_lo) / 2
    ch_slope_pct = (cur_mid - prev_mid) / c * 100
    return {
        "lower_wick": round(lower_wick, 1),
        "upper_wick": round(upper_wick, 1),
        "body": round(body, 1),
        "ch_pos": round(ch_pos, 1),
        "ch_hi": round(ch_hi, 2),
        "ch_lo": round(ch_lo, 2),
        "ch_width_pct": round(ch_width_pct, 3),
        "ch_slope_pct": round(ch_slope_pct, 3),
        "ch_lookback": lookback,
    }


def read_magnitude(rows, spot, atm_iv, K=2.5356):
    """Forecast the REST-OF-DAY move size (magnitude is forecastable; sign is not).

    Two estimates of the expected |move| over the remaining session:
      - realized-vol model: K * Garman-Klass(open->now). GK morning vol predicts
        afternoon |move| with OOS Spearman IC ~0.4 (0.21 OOS); K=2.54 calibrated
        on 359 days to the 11:30 split.
      - options-implied: atm_iv * sqrt(remaining trading-time).
    realized > implied => options underpricing the move ('cheap'); else 'rich'.
    """
    import math

    if len(rows) < 3:
        return {}
    terms = []
    for _m, o, h, low, c, _v in rows:
        if o > 0 and low > 0 and h > 0:
            terms.append(
                0.5 * math.log(h / low) ** 2
                - (2 * math.log(2) - 1) * math.log(max(c / o, 1e-9)) ** 2
            )
    if not terms:
        return {}
    gk = math.sqrt(max(sum(terms) / len(terms), 0.0)) * 100  # % per-bar vol
    minute_now = rows[-1][0]
    remaining_frac = max(390 - minute_now, 5) / 390.0  # fraction of session left
    realized_pct = K * gk
    implied_pct = (atm_iv or 0) * math.sqrt(remaining_frac / 252.0) * 100
    return {
        "gk_vol": round(gk, 3),
        "exp_move_realized_pct": round(realized_pct, 3),
        "exp_move_realized_usd": round(spot * realized_pct / 100, 2),
        "exp_move_implied_pct": round(implied_pct, 3),
        "exp_move_implied_usd": round(spot * implied_pct / 100, 2),
        "premium_read": "cheap" if realized_pct > implied_pct else "rich",
        "remaining_frac": round(remaining_frac, 3),
    }


def write_signal(
    pa,
    op,
    gp,
    gcard,
    signal_ts,
    setups=None,
    micro=None,
    magnitude=None,
    permission=None,
    volatility_structure=None,
    target_plan=None,
    decision_receipt=None,
    permission_score_trend=None,
    edge_token_position=None,
    regime_refinement=None,
    source_freshness=None,
    reference_levels=None,
    level_states=None,
    timeframe_agreement=None,
    transition_pressure=None,
):
    """Drop a machine-readable signal.json the trade_intent pipeline can read."""

    def rounded(value, digits):
        return round(value, digits) if isinstance(value, (int, float)) else None

    volatility_structure = volatility_structure or {}
    reference_levels = reference_levels or {}
    level_states = level_states or {}
    source_freshness = {
        "signal_generated_at": signal_ts,
        **(source_freshness or {}),
    }
    entry_setup = primary_trade_setup(setups)
    context_setup = primary_context_setup(setups)
    effective_entry_gate = (
        ((permission or {}).get("setup_conviction") or {}).get("entry_gate")
        or (decision_receipt or {}).get("entry_gate")
        or {}
    )
    effective_context_gate = (
        ((permission or {}).get("setup_conviction") or {}).get("context_gate")
        or (decision_receipt or {}).get("context_gate")
        or {}
    )
    sig = {
        "schema": "sharpedge.signal.v1",
        "ts": signal_ts,
        "symbol": "SPY",
        "spot": round(pa["spot"], 2),
        "day_chg": round(pa["day_chg"], 3),
        "vwap": round(pa["vwap"], 2),
        "vs_vwap": round(pa["vs_vwap"], 3),
        "balance_high": round(pa["balance_high"], 2),
        "balance_low": round(pa["balance_low"], 2),
        "position_in_balance": round(pa["position_in_balance"], 3),
        "balance_state": pa["balance_state"],
        "balance_label": pa["balance_label"],
        "balance_width_pct": round(pa["balance_width_pct"], 3),
        "balance_window_bars": pa["balance_window_bars"],
        "balance_reference": pa["balance_reference"],
        "dominant_balance_name": pa["dominant_balance_name"],
        "dominant_balance_reason": pa["dominant_balance_reason"],
        "dominant_balance_previous_name": pa["dominant_balance_previous_name"],
        "dominant_balance_flip": pa["dominant_balance_flip"],
        "balance_models": pa["balance_models"],
        "balance_confluence": pa["balance_confluence"],
        "balance_disagreement": pa["balance_disagreement"],
        "session_position_in_range": round(pa["session_position_in_range"], 3),
        "rng_pos": round(pa["rng_pos"], 1),
        "mom15": round(pa["mom15"], 3),
        "vol_mult": round(pa["vol_mult"], 2),
        "volume_profile": pa.get("volume_profile") or {},
        "call_wall": op.get("call_wall"),
        "put_wall": op.get("put_wall"),
        "call_volume_wall": op.get("call_volume_wall"),
        "put_volume_wall": op.get("put_volume_wall"),
        "pcr": round(op.get("pcr", 0), 2),
        "pcvr": round(op.get("pcvr", 0), 2),
        "call_volume_total": rounded(op.get("call_volume_total"), 0),
        "put_volume_total": rounded(op.get("put_volume_total"), 0),
        "atm_strike": op.get("atm_strike"),
        "atm_iv": round(op.get("atm_iv", 0), 4),
        "atm_call_iv": rounded(op.get("atm_call_iv"), 4),
        "atm_put_iv": rounded(op.get("atm_put_iv"), 4),
        "atm_iv_skew": rounded(op.get("atm_iv_skew"), 4),
        "atm_call_delta": rounded(op.get("atm_call_delta"), 4),
        "atm_put_delta": rounded(op.get("atm_put_delta"), 4),
        "atm_call_theta": rounded(op.get("atm_call_theta"), 4),
        "atm_put_theta": rounded(op.get("atm_put_theta"), 4),
        "atm_call_vega": rounded(op.get("atm_call_vega"), 4),
        "atm_put_vega": rounded(op.get("atm_put_vega"), 4),
        "atm_call_rho": rounded(op.get("atm_call_rho"), 4),
        "atm_put_rho": rounded(op.get("atm_put_rho"), 4),
        "atm_call_theo": rounded(op.get("atm_call_theo"), 2),
        "atm_put_theo": rounded(op.get("atm_put_theo"), 2),
        "atm_call_last_trade_price": rounded(op.get("atm_call_last_trade_price"), 2),
        "atm_put_last_trade_price": rounded(op.get("atm_put_last_trade_price"), 2),
        "atm_call_bid": rounded(op.get("atm_call_bid"), 2),
        "atm_call_ask": rounded(op.get("atm_call_ask"), 2),
        "atm_put_bid": rounded(op.get("atm_put_bid"), 2),
        "atm_put_ask": rounded(op.get("atm_put_ask"), 2),
        "atm_call_spread": rounded(op.get("atm_call_spread"), 2),
        "atm_put_spread": rounded(op.get("atm_put_spread"), 2),
        "atm_call_spread_pct": rounded(op.get("atm_call_spread_pct"), 4),
        "atm_put_spread_pct": rounded(op.get("atm_put_spread_pct"), 4),
        "atm_straddle_mid": rounded(op.get("atm_straddle_mid"), 2),
        "exp": op.get("exp"),
        "gamma_regime": gp.get("regime"),
        "pin": gp.get("pin"),
        "max_pain": gp.get("max_pain"),
        "setup_tag": gcard["tag"] if gcard else None,
        "setup_bias": gcard["bias"] if gcard else None,
        "entry_setup_tag": effective_entry_gate.get("tag") or entry_setup.get("tag"),
        "entry_setup_bias": effective_entry_gate.get("bias") or entry_setup.get("bias"),
        "context_setup_tag": effective_context_gate.get("tag")
        or context_setup.get("tag"),
        "context_setup_bias": effective_context_gate.get("bias")
        or context_setup.get("bias"),
        "setup_cards": setups or [],
        "reference_levels": reference_levels,
        "level_states": level_states,
        "execution_structure_state": (permission or {}).get("structure_state") or {},
        "execution_acceptance_state": (permission or {}).get("acceptance_state") or {},
        "execution_location_state": (permission or {}).get("location_state") or {},
        "execution_dealer_state": (permission or {}).get("dealer_state") or {},
        "execution_volume_state": (permission or {}).get("volume_state") or {},
        "execution_trend_state": (permission or {}).get("trend_state") or {},
        "execution_time_state": (permission or {}).get("time_state") or {},
        "volatility_state": volatility_structure.get("volatility_state"),
        "structure_state": volatility_structure.get("structure_state"),
        "volatility_structure": volatility_structure,
        "micro": micro or {},
        "magnitude": magnitude or {},
        "trade_permission": permission or {},
        "target_plan": target_plan or {},
        "entry_gate": (decision_receipt or {}).get("entry_gate") or {},
        "context_gate": (decision_receipt or {}).get("context_gate") or {},
        "decision_receipt": decision_receipt or {},
        "permission_score_trend": permission_score_trend or {},
        "edge_token_position": edge_token_position or {},
        "regime_refinement": regime_refinement or {},
        "weekly_context": pa.get("weekly_context") or {},
        "monthly_context": pa.get("monthly_context") or {},
        "timeframe_agreement": timeframe_agreement or {},
        "transition_pressure": transition_pressure or {},
        "source_freshness": source_freshness,
    }
    out = os.path.expanduser("~/SharpEdge-System/outputs")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "signal.json"), "w") as f:
        json.dump(sig, f, indent=2)


def main():
    rows, price_source = fetch_intraday_with_source()
    _spot_opt, book, options_source = fetch_options_with_source()
    pa = read_price_action(rows)
    op = read_options(pa["spot"], book)
    lines = synthesize(pa, op)
    levels = reference_levels(rows)
    level_states = build_level_state_map(rows, levels)
    volatility_structure = read_volatility_structure(rows, pa)
    gp = gamma_profile(book, pa["spot"])
    gcard = gamma_card(gp)

    force_cards = []
    continuation = detect_negative_gamma_continuation(pa, op, gp, bars=rows)
    if continuation:
        force_cards.append(continuation)
    sticky_noise = detect_sticky_noise(pa, op, gp)
    if sticky_noise:
        force_cards.append(sticky_noise)

    setups = (
        force_cards + detect_failed_breaks(rows, levels) + detect_exhaustion(rows, pa)
    )
    coil = detect_volatility_coil(rows, pa, volatility_structure)
    if coil:
        setups.append(coil)
    if gcard:
        setups = [gcard] + setups  # gamma regime sits at the very top
    micro = read_microstructure(rows)
    magnitude = read_magnitude(rows, pa["spot"], op.get("atm_iv", 0))
    context_attachment = build_context_attachment(rows, spot=pa["spot"])
    weekly_context = context_attachment["weekly_context"]
    monthly_context = context_attachment["monthly_context"]
    weekly_context_rows = context_attachment["weekly_rows"]
    monthly_context_rows = context_attachment["monthly_rows"]
    carry_levels = context_attachment["carry_levels"]
    monthly_levels = context_attachment["monthly_levels"]
    pa["weekly_context"] = weekly_context
    pa["monthly_context"] = monthly_context
    permission = score_trade_permission(
        rows, pa, levels, setups, op, gp, magnitude, volatility_structure
    )
    timeframe_agreement = build_timeframe_agreement(
        pa,
        weekly_context,
        monthly_context_rows,
        permission,
    )
    target_plan = infer_target(pa, op, permission, gp, micro, magnitude, setups)
    signal_ts = dt.datetime.now().isoformat()
    out_dir = os.path.expanduser("~/SharpEdge-System/outputs")
    receipt_path = os.path.join(out_dir, "permission_receipts_spy.jsonl")
    prior_receipts = load_recent_receipts(Path(receipt_path))
    decision_receipt = build_decision_receipt(
        signal_ts,
        "SPY",
        pa.get("spot"),
        permission,
        target_plan,
        setups,
        prior_receipts[-1] if prior_receipts else None,
    )
    annotate_setup_conviction(permission, decision_receipt.get("setup_events") or [])
    decision_receipt["setup_conviction"] = permission.get("setup_conviction") or {}
    decision_receipt["setup"] = (permission.get("setup_conviction") or {}).get(
        "setup_tag"
    ) or decision_receipt.get("setup")
    decision_receipt["setup_bias"] = (
        ((permission.get("setup_conviction") or {}).get("entry_gate") or {}).get("bias")
    ) or decision_receipt.get("setup_bias")
    decision_receipt["entry_gate"] = (
        ((permission.get("setup_conviction") or {}).get("entry_gate"))
        or decision_receipt.get("entry_gate")
        or {}
    )
    decision_receipt["primary_setup_event"] = primary_setup_event(
        decision_receipt.get("setup_events") or [], decision_receipt.get("setup")
    )
    permission_score_trend = build_permission_score_trend(
        decision_receipt, prior_receipts
    )
    transition_pressure = build_transition_pressure_packet(
        pa,
        op,
        gp,
        volatility_structure,
        setups,
        decision_receipt,
        prior_receipts,
        level_states=level_states,
    )
    edge_token_position = {}
    regime_refinement = annotate_market_behavior(
        pa,
        op,
        gp,
        permission,
        target_plan,
        magnitude,
        setups,
        edge_token_position,
    )
    append_decision_receipt(Path(receipt_path), decision_receipt)
    write_signal(
        pa,
        op,
        gp,
        gcard,
        signal_ts,
        setups,
        micro,
        magnitude,
        permission,
        volatility_structure,
        target_plan,
        decision_receipt,
        permission_score_trend,
        edge_token_position,
        regime_refinement,
        {
            "price": price_source,
            "options": options_source,
        },
        levels,
        level_states,
        timeframe_agreement,
        transition_pressure,
    )
    write_ace_snapshot(rows, pa, levels, op, gp, out_dir)
    with open(f"{OUT_DIR}/cockpit_chart.svg", "w") as f:
        f.write(
            chart_svg(
                rows,
                pa,
                levels,
                setups,
                volatility_structure,
                level_states=level_states,
            )
        )
    with open(f"{OUT_DIR}/cockpit_weekly_context.svg", "w") as f:
        f.write(
            build_weekly_context_svg(
                weekly_context_rows,
                carry_levels,
                symbol="SPY",
                lookback_days=5,
            )
        )
    with open(f"{OUT_DIR}/cockpit_monthly_context.svg", "w") as f:
        f.write(
            build_monthly_context_svg(
                monthly_context_rows,
                monthly_levels,
                symbol="SPY",
                lookback_months=6,
            )
        )
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    with open(f"{OUT_DIR}/cockpit.html", "w") as f:
        f.write(
            render_live_read_html(
                pa,
                op,
                lines,
                setups,
                permission,
                micro,
                magnitude,
                gp,
                permission_score_trend,
                edge_token_position,
                regime_refinement,
                weekly_context,
                monthly_context,
                stamp,
                level_states=level_states,
                timeframe_agreement=timeframe_agreement,
                transition_pressure=transition_pressure,
            )
        )
    with open(f"{OUT_DIR}/runner_handoff_live.html", "w") as f:
        f.write(
            render_runner_handoff_live_html(
                pa,
                op,
                lines,
                setups,
                permission,
                micro,
                magnitude,
                gp,
                permission_score_trend,
                edge_token_position,
                regime_refinement,
                weekly_context,
                monthly_context,
                stamp,
                decision_receipt.get("setup_events") or [],
                timeframe_agreement=timeframe_agreement,
                transition_pressure=transition_pressure,
            )
        )
    print(
        f"spot ${pa['spot']:.2f} | day {pa['day_chg']:+.2f}% | "
        f"vs VWAP {pa['vs_vwap']:+.2f}% | balance {pa['position_in_balance']:.2f} "
        f"({pa['balance_state']}) | rng {pa['rng_pos']:.0f}% | vol {pa['vol_mult']:.1f}x"
    )
    levels_str = " ".join(f"{k}=${v:.2f}" for k, v in levels.items())
    print(f"  levels: {levels_str}")
    setup_conviction = permission.get("setup_conviction") or {}
    print(
        f"  setup conviction: {setup_conviction.get('setup_gate', 'NONE')} "
        f"{setup_conviction.get('setup_conviction_score', 0)}/100 "
        f"bias={setup_conviction.get('bias', 'NEUTRAL')}"
    )
    print(
        f"  execution gate: {permission['trade_gate']} "
        f"{permission['trade_permission_score']}/100 "
        f"bias={permission['bias']}"
    )
    print(
        f"  authority engine: {permission.get('authority_engine', 'legacy')} "
        f"mode={permission.get('authority_mode', 'full_contract')}"
    )
    print(
        f"  transition pressure: {transition_pressure.get('transition_state', 'unknown')} "
        f"{transition_pressure.get('transition_pressure_score', 0)}/100 "
        f"attention={transition_pressure.get('attention_state', 'watch')}"
    )
    spine = permission.get("bucket_conditioned_spine") or {}
    print(
        f"  spine authority: {spine.get('gate', 'BLOCK')} "
        f"{spine.get('score', permission['trade_permission_score'])}/100 "
        f"action={spine.get('recommended_action', 'watch_only')}"
    )
    print(
        f"  behavior: {regime_refinement.get('primary_behavior')} -> "
        f"{regime_refinement.get('behavior_summary')}"
    )
    if setups:
        for s in setups:
            print(f"  >> {s['tag']} -> {s['bias']}: {s['detail']}")
    else:
        print("  >> no failed-break/exhaustion/compression setup right now")
    for t, k, d in lines:
        print(f"  [{k:4}] {t}: {d}")


if __name__ == "__main__":
    main()
