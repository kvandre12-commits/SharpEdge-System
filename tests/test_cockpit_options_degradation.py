from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

COCKPIT_DIR = Path(__file__).resolve().parents[1] / "cockpit"
if str(COCKPIT_DIR) not in sys.path:
    sys.path.insert(0, str(COCKPIT_DIR))

import market_data_sources as sources  # noqa: E402
import options_availability  # noqa: E402


class OptionsDegradationTests(unittest.TestCase):
    def test_empty_book_is_explicitly_unavailable(self) -> None:
        surface = sources.read_options_surface(500.0, {})

        self.assertFalse(surface["available"])
        self.assertIsNone(surface["call_wall"])
        self.assertIsNone(surface["put_wall"])

    def test_fail_soft_fetch_preserves_failure_provenance(self) -> None:
        with patch.object(
            options_availability,
            "fetch_cboe_options_book",
            side_effect=RuntimeError("provider denied request"),
        ):
            spot, book, source = options_availability.fetch_cboe_options_book_fail_soft("SPY")

        self.assertEqual(spot, 0.0)
        self.assertEqual(book, {})
        self.assertEqual(source["provider"], "cboe")
        self.assertEqual(source["symbol"], "SPY")
        self.assertFalse(source["available"])
        self.assertEqual(source["error"], "RuntimeError: provider denied request")

    def test_fail_soft_fetch_passes_success_through(self) -> None:
        expected = (501.25, {"expiry": {}}, {"provider": "cboe", "available": True})
        with patch.object(
            options_availability, "fetch_cboe_options_book", return_value=expected
        ) as fetch:
            actual = options_availability.fetch_cboe_options_book_fail_soft("SPY", timeout=7)

        self.assertEqual(actual, expected)
        fetch.assert_called_once_with("SPY", timeout=7)


if __name__ == "__main__":
    unittest.main()
