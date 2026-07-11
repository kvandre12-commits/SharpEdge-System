from __future__ import annotations

import unittest
from unittest.mock import patch

import scripts.send_trade_card_to_discord as discord_sender


class SendTradeCardToDiscordTests(unittest.TestCase):
    def test_post_requires_webhook(self) -> None:
        with patch.object(discord_sender, "WEBHOOK", ""):
            with self.assertRaisesRegex(RuntimeError, "DISCORD_WEBHOOK_URL missing"):
                discord_sender._post("hello")

    def test_send_splits_large_messages(self) -> None:
        posted: list[str] = []
        huge = ("x" * 1200) + "\n" + ("y" * 1200)

        with patch.object(discord_sender, "_post", side_effect=posted.append):
            discord_sender.send(huge)

        self.assertEqual(len(posted), 2)
        self.assertTrue(posted[0].startswith("**(part 1/2)**\n"))
        self.assertTrue(posted[1].startswith("**(part 2/2)**\n"))


if __name__ == "__main__":
    unittest.main()
