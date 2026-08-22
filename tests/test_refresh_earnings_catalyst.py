from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cockpit"))

import refresh_earnings_catalyst as run

_REPORT = {
    "generated": "2026-08-21",
    "tickers": {},
    "needs_attention": ["TSLA"],
    "suggestions": {"MSFT": "2026-10-28"},
    "mismatches": [],
    "new_filings": ["AMD"],
    "current_accessions": {"AMD": "acc-NEW", "AAPL": "acc-1"},
}


def test_load_prev_accessions_missing_file_returns_empty(tmp_path):
    assert run.load_prev_accessions(str(tmp_path / "nope.json")) == {}


def test_save_then_load_round_trip(tmp_path):
    path = str(tmp_path / "nested" / "state.json")
    run.save_accessions(path, {"AMD": "acc-9"})
    assert run.load_prev_accessions(path) == {"AMD": "acc-9"}


def test_load_prev_accessions_tolerates_garbage(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert run.load_prev_accessions(str(path)) == {}


def test_main_prints_summary_and_persists_state(tmp_path, capsys):
    state = str(tmp_path / "state.json")
    with patch.object(run, "build_earnings_catalyst_report", return_value=_REPORT):
        rc = run.main(["--state", state])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUGGEST MSFT" in out
    assert "NEW SEC FILING" in out
    # state persisted with current accessions for next run's newness diff
    saved = json.loads(Path(state).read_text(encoding="utf-8"))
    assert saved["accessions"] == {"AMD": "acc-NEW", "AAPL": "acc-1"}


def test_main_json_mode_emits_full_report(tmp_path, capsys):
    with patch.object(run, "build_earnings_catalyst_report", return_value=_REPORT):
        run.main(["--json", "--no-state"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["suggestions"] == {"MSFT": "2026-10-28"}


def test_main_no_state_does_not_write(tmp_path):
    state = tmp_path / "state.json"
    with patch.object(run, "build_earnings_catalyst_report", return_value=_REPORT):
        run.main(["--no-state", "--state", str(state)])
    assert not state.exists()
