#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from scripts.agents.overnight_carry_data import load_gap_history, load_regime_context
    from scripts.agents.overnight_carry_logic import build_payload
    from scripts.agents.overnight_carry_view import write_outputs
except ModuleNotFoundError:  # pragma: no cover
    from overnight_carry_data import load_gap_history, load_regime_context
    from overnight_carry_logic import build_payload
    from overnight_carry_view import write_outputs


OUTDIR = Path("outputs")
PRESET_FILTERS = {
    "failed_breakdown": {"event_type": "FAILED_BREAKDOWN"},
    "clean_breakdown": {"event_type": "CLEAN_BREAKDOWN"},
    "range_compression": {"event_type": "RANGE_COMPRESSION"},
    "high_vol": {"vol_state": "high"},
}


def _conditioning_filters(args: argparse.Namespace) -> dict[str, Any]:
    raw = {
        "event_type": args.condition_event_type,
        "open_regime_label": args.condition_open_regime_label,
        "liquidity_regime_type": args.condition_liquidity_regime_type,
        "setup_dir": args.condition_setup_dir,
        "key_source": args.condition_key_source,
        "vol_state": args.condition_vol_state,
        "vol_trend_state": args.condition_vol_trend_state,
        "dp_state": args.condition_dp_state,
        "macro_state": args.condition_macro_state,
        "regime_label": args.condition_regime_label,
        "failed_breakdown_open": args.condition_failed_breakdown_open,
        "accepted_breakdown_open": args.condition_accepted_breakdown_open,
    }
    return {key: value for key, value in raw.items() if value not in (None, "")}



def _comparison_contexts(
    args: argparse.Namespace, primary_filters: dict[str, Any]
) -> list[dict[str, Any]]:
    contexts = []
    names = [name.strip() for name in str(args.compare_presets or "").split(",") if name.strip()]
    for name in names:
        filters = PRESET_FILTERS.get(name)
        if not filters or filters == primary_filters:
            continue
        context = load_regime_context(
            proxy_symbol=args.context_proxy_symbol,
            filters=filters,
            db_path=args.context_db_path,
            years=args.history_years,
        )
        context["label"] = name
        contexts.append(context)
    return contexts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SharpEdge overnight carry brief.")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--option-type", default="call", choices=["call", "put"])
    parser.add_argument("--expiration", required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--iv", type=float, required=True)
    parser.add_argument("--vega", type=float, default=None)
    parser.add_argument("--close-timestamp", default="")
    parser.add_argument("--close-to-open-hours", type=float, default=17.5)
    parser.add_argument("--history-years", type=int, default=10)
    parser.add_argument("--db-path", default="data/market_data.db")
    parser.add_argument("--disable-yfinance-fallback", action="store_true")
    parser.add_argument("--context-proxy-symbol", default="SPY")
    parser.add_argument("--context-db-path", default="data/spy_truth.db")
    parser.add_argument("--condition-event-type", default="")
    parser.add_argument("--condition-open-regime-label", default="")
    parser.add_argument("--condition-liquidity-regime-type", default="")
    parser.add_argument("--condition-setup-dir", default="")
    parser.add_argument("--condition-key-source", default="")
    parser.add_argument("--condition-vol-state", default="")
    parser.add_argument("--condition-vol-trend-state", default="")
    parser.add_argument("--condition-dp-state", default="")
    parser.add_argument("--condition-macro-state", default="")
    parser.add_argument("--condition-regime-label", default="")
    parser.add_argument("--condition-failed-breakdown-open", type=int, choices=[0, 1], default=None)
    parser.add_argument("--condition-accepted-breakdown-open", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--compare-presets",
        default="failed_breakdown,clean_breakdown,range_compression,high_vol",
        help="Comma-separated preset comparison rack.",
    )
    parser.add_argument("--output-base", default="")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = {
        "symbol": args.symbol,
        "spot": args.spot,
        "strike": args.strike,
        "option_type": args.option_type,
        "expiration": args.expiration,
        "delta": args.delta,
        "gamma": args.gamma,
        "theta": args.theta,
        "iv": args.iv,
        "vega": args.vega,
        "close_timestamp": args.close_timestamp,
        "close_to_open_hours": args.close_to_open_hours,
    }
    gap_history = load_gap_history(
        args.symbol,
        db_path=args.db_path,
        years=args.history_years,
        allow_yfinance_fallback=not args.disable_yfinance_fallback,
    )
    primary_filters = _conditioning_filters(args)
    conditioning_context = load_regime_context(
        proxy_symbol=args.context_proxy_symbol,
        filters=primary_filters,
        db_path=args.context_db_path,
        years=args.history_years,
    )
    conditioning_context["label"] = "conditioned"
    payload = build_payload(
        contract,
        gap_history,
        conditioning_context,
        _comparison_contexts(args, primary_filters),
    )
    output_base = (
        Path(args.output_base)
        if args.output_base
        else OUTDIR / f"{str(args.symbol).lower()}_overnight_carry"
    )
    json_path, txt_path = write_outputs(payload, output_base)
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
