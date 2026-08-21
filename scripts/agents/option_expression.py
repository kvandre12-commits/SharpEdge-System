#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.agents.option_expression_logic import build_branch_greek_dollar_plan
    from scripts.agents.option_expression_view import write_outputs
except ModuleNotFoundError:  # pragma: no cover
    from option_expression_logic import build_branch_greek_dollar_plan
    from option_expression_view import write_outputs

OUTDIR = Path('outputs')
SIGNAL_JSON = OUTDIR / 'signal.json'
POSITION_LAB_JSON = OUTDIR / 'spy_position_lab.json'
APPROVAL_JSON = OUTDIR / 'approval_decision.json'



def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))



def _utc_now() -> str:
    return datetime.now(UTC).isoformat()



def _expression_objective(branch: dict[str, Any], geometry: dict[str, Any]) -> str:
    family = str(branch.get('structure_family') or 'unknown')
    premium = str(geometry.get('premium_read') or 'unknown')
    if family == 'call_debit_spread':
        return f'defined-risk upside participation without paying for naked {premium} premium'
    if family == 'put_debit_spread':
        return 'defined-risk downside participation with capped loss and cleaner failure map'
    if family == 'long_put':
        return 'single-leg downside convexity because no clean short hedge leg is nearby'
    if family == 'long_call':
        return 'single-leg upside convexity because the nearby spread lane is not clean'
    if family == 'no_forced_position':
        return 'wait while geometry decides; preserve capital instead of forcing premium exposure'
    return 'translate the market thesis into a defined, explainable options expression'



def build_payload(
    signal: dict[str, Any], position_lab: dict[str, Any], approval: dict[str, Any]
) -> dict[str, Any]:
    geometry = position_lab.get('geometry') or {}
    branches = []
    for branch in position_lab.get('branches') or []:
        plan = branch.get('greek_dollar_plan') or build_branch_greek_dollar_plan(branch, geometry)
        branches.append(
            {
                'branch_id': branch.get('branch_id'),
                'direction': branch.get('direction'),
                'status': branch.get('status'),
                'structure_family': branch.get('structure_family'),
                'structure_label': branch.get('structure_label'),
                'expression_objective': _expression_objective(branch, geometry),
                'trigger': branch.get('trigger'),
                'invalidation': branch.get('invalidation'),
                'thesis': branch.get('thesis'),
                'caution': branch.get('caution'),
                'greek_dollar_plan': plan,
            }
        )
    return {
        'schema': 'sharpedge.option_expression.v1',
        'generated_at_utc': _utc_now(),
        'symbol': position_lab.get('symbol') or signal.get('symbol') or 'SPY',
        'source_artifacts': {
            'signal': str(SIGNAL_JSON),
            'position_lab': str(POSITION_LAB_JSON),
            'approval': str(APPROVAL_JSON),
            'snapshot': ((position_lab.get('source_artifacts') or {}).get('snapshot')),
        },
        'expression_doctrine': {
            'core_rule': "If you can't say 'I lose $X if Y happens,' you're not trading — you're donating.",
            'thinking_order': [
                'market_hypothesis',
                'greek_dollar_plan',
                'structure_family',
                'contract_realization',
            ],
            'operator_note': 'Strikes are last-mile outputs; Greek-dollar consequences are the actual thinking layer.',
        },
        'market_hypothesis': {
            'setup_tag': geometry.get('setup_tag'),
            'gamma_regime': geometry.get('gamma_regime'),
            'dealer_state': geometry.get('dealer_state'),
            'premium_read': geometry.get('premium_read'),
            'spot': geometry.get('spot'),
            'vwap': geometry.get('vwap'),
            'pin': geometry.get('pin'),
            'call_wall': geometry.get('call_wall'),
            'put_wall': geometry.get('put_wall'),
            'exp_move_implied_usd': geometry.get('exp_move_implied_usd'),
        },
        'branch_expressions': branches,
        'execution_boundary': {
            'trade_allowed': bool(approval.get('trade_allowed')),
            'broker_order_allowed': bool(approval.get('broker_order_allowed')),
            'decision': approval.get('decision'),
            'blocking_reasons': list(approval.get('blocking_reasons') or [])[:5],
        },
    }



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build SharpEdge option expression doctrine.')
    parser.add_argument('--symbol', default='SPY')
    parser.add_argument('--output-base', default='')
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal = _read_json(SIGNAL_JSON)
    position_lab = _read_json(POSITION_LAB_JSON)
    approval = _read_json(APPROVAL_JSON)
    payload = build_payload(signal, position_lab, approval)
    base = Path(args.output_base) if args.output_base else OUTDIR / f"{str(args.symbol).lower()}_option_expression"
    json_path, txt_path = write_outputs(payload, base)
    print(f'wrote {json_path}')
    print(f'wrote {txt_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
