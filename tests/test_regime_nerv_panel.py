from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from regime_nerv_panel import BoardSource, render_panel_html, write_surfaces  # noqa: E402


def test_render_panel_includes_structure_taxonomy(tmp_path: Path) -> None:
    board = tmp_path / "ctc_nerv_trade_desk.json"
    board.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-26T22:30:00+00:00",
                "summary": {
                    "row_count": 1,
                    "manual_validate_count": 0,
                    "states": {"refresh_quote_required": 1},
                },
                "rows": [
                    {
                        "rank": 1,
                        "ticker": "WMT",
                        "company": "Walmart Inc.",
                        "desk_state": "refresh_quote_required",
                        "structure_family": "branch_defined_debit_spread",
                        "structure_complexity": "branch_pending",
                        "nerv_score": 49.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_panel_html([BoardSource("Cartridge: WMT", board)])

    assert "Cartridge: WMT" in html
    assert "branch_defined_debit_spread" in html
    assert "branch_pending" in html
    assert "Walmart Inc." in html


def test_write_surfaces_creates_panel_and_split(tmp_path: Path) -> None:
    paths = write_surfaces(cockpit_dir=tmp_path, refresh_seconds=10, sources=[])

    panel = Path(paths["panel"])
    split = Path(paths["split"])
    assert panel.exists()
    assert split.exists()
    assert "No Regime/NERV board found" in panel.read_text(encoding="utf-8")
    assert "cockpit.html" in split.read_text(encoding="utf-8")
