#!/usr/bin/env python3
"""Generic regime-cartridge alias for the CTC/NERV trade-desk engine.

This exists so non-CTC company/institution workbooks can use the same taxonomy
without pretending every future cartridge is literally named CTC. Naming matters;
copy-paste mythology is how systems get haunted.
"""

from __future__ import annotations

from ctc_nerv_trade_desk import main


if __name__ == "__main__":
    raise SystemExit(main())
