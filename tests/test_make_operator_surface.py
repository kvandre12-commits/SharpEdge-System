from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from make_operator_surface import _execution_vector_interactions_card  # noqa: E402


def test_operator_surface_renders_vector_interactions_card():
    signal = {
        "trade_permission": {
            "execution_vector_interactions": {
                "summary": {
                    "interaction_balance": "mixed",
                    "favorable_count": 2,
                    "warning_count": 1,
                    "strong_favorable_count": 1,
                    "strong_warning_count": 1,
                },
                "best": [
                    {
                        "interaction_id": "trend_acceptance_alignment",
                        "classification": "strongly_good",
                        "label": "Trend + acceptance aligned",
                        "reason": "Directional drive and acceptance are working together.",
                    }
                ],
                "warnings": [
                    {
                        "interaction_id": "trend_volume_conflict",
                        "classification": "strongly_bad",
                        "label": "Trend without participation",
                        "reason": "Tape is moving but volume is not backing it.",
                    }
                ],
            }
        }
    }

    html = _execution_vector_interactions_card(signal)

    assert "vector interactions" in html.lower()
    assert "interaction balance" in html
    assert "Trend + acceptance aligned" in html
    assert "Trend without participation" in html
    assert "strongly good" in html
    assert "strongly bad" in html
