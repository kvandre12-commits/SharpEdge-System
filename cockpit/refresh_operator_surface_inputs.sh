#!/data/data/com.termux/files/usr/bin/bash
# Refresh the operator-surface upstream artifacts from current local inputs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export AGENT_LANGUAGE_AGENT_ID="${AGENT_LANGUAGE_AGENT_ID:-code-puppy-fd03ba}"
POSITION_LAB_SYMBOL="${SYMBOL:-SPY}"
POSITION_LAB_NERV_MAX_EXPIRATIONS="${POSITION_LAB_NERV_MAX_EXPIRATIONS:-20}"

python3 scripts/build_robinhood_fvg_monitor.py
python3 -m scripts.agents.nerv_curator
python3 -m scripts.agents.controller_agent
python3 -m scripts.agents.agent_v1_decision
python3 -m scripts.agents.trade_journal_hints
python3 -m scripts.agents.operator_brief
python3 -m scripts.agents.operator_session_review
python3 -m scripts.agents.agent_language_objects
python3 -m scripts.agents.robinhood_beta_execution
python3 -m scripts.agents.morning_open_dashboard
python3 scripts/nerv_free_data_adapter.py \
  --symbols "$POSITION_LAB_SYMBOL" \
  --max-expirations "$POSITION_LAB_NERV_MAX_EXPIRATIONS" \
  --board-limit 0 \
  --output-dir outputs/nerv_position_lab \
  --retention-hours 24 \
  --skip-panel-refresh \
  --skip-curator-refresh
python3 -m scripts.agents.position_lab --symbol "$POSITION_LAB_SYMBOL"
python3 -m scripts.agents.option_expression --symbol "$POSITION_LAB_SYMBOL"
python3 -m scripts.agents.operator_decision_card
