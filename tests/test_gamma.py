from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from gamma import gamma_card, gamma_profile


def test_gamma_profile_marks_missing_gamma_as_unknown_instead_of_fake_pin():
    today = dt.date.today()
    book = {
        today: {
            500.0: {"C": {"gamma": 0.0, "open_interest": 10}, "P": {}},
            745.0: {
                "C": {"gamma": 0.0, "open_interest": 8840},
                "P": {"gamma": 0.0, "open_interest": 8274},
            },
            750.0: {
                "C": {"gamma": 0.0, "open_interest": 13535},
                "P": {"gamma": 0.0, "open_interest": 3757},
            },
        }
    }

    profile = gamma_profile(book, 747.0)

    assert profile["regime"] == "unknown"
    assert profile["pin"] is None
    assert profile["pin_dist"] is None
    assert profile["net_gamma"] is None
    assert profile["gamma_data_quality"] == "missing"


def test_gamma_card_skips_unknown_gamma_profile():
    gp = {
        "exp": dt.date.today().isoformat(),
        "dte": 0,
        "regime": "unknown",
        "net_gamma": None,
        "pin": None,
        "pin_dist": None,
        "max_pain": 745.0,
        "spot": 747.0,
        "gamma_data_quality": "missing",
    }

    assert gamma_card(gp) is None


def test_gamma_profile_uses_informative_gamma_to_find_nearby_pin():
    today = dt.date.today()
    book = {
        today: {
            745.0: {
                "C": {"gamma": 0.0772, "open_interest": 8840},
                "P": {"gamma": 0.0772, "open_interest": 8274},
            },
            750.0: {
                "C": {"gamma": 0.0704, "open_interest": 13535},
                "P": {"gamma": 0.0704, "open_interest": 3757},
            },
        }
    }

    profile = gamma_profile(book, 746.85)

    assert profile["regime"] == "positive"
    assert profile["pin"] == 745.0
    assert profile["gamma_data_quality"] == "ok"
    assert profile["net_gamma"] is not None


def test_gamma_profile_rejects_expired_only_chain():
    expired = dt.date.today() - dt.timedelta(days=1)
    book = {
        expired: {
            745.0: {
                "C": {"gamma": 0.08, "open_interest": 1000},
                "P": {"gamma": 0.07, "open_interest": 900},
            }
        }
    }

    profile = gamma_profile(book, 745.0)

    assert profile["regime"] == "unknown"
    assert profile["gamma_data_quality"] == "expired"
    assert profile["dte"] == -1
    assert profile["pin"] is None
    assert gamma_card(profile) is None
