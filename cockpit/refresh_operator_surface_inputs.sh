#!/data/data/com.termux/files/usr/bin/bash
# Refresh the operator-surface upstream artifacts from current local inputs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export AGENT_LANGUAGE_AGENT_ID="${AGENT_LANGUAGE_AGENT_ID:-code-puppy-6fbf30}"

python3 scripts/build_robinhood_fvg_monitor.py
python3 -m scripts.agents.controller_agent
python3 -m scripts.agents.agent_v1_decision
python3 -m scripts.agents.trade_journal_hints
python3 -m scripts.agents.operator_brief
python3 -m scripts.agents.operator_session_review
python3 -m scripts.agents.agent_language_objects
python3 -m scripts.agents.robinhood_beta_execution
python3 -m scripts.agents.morning_open_dashboard
