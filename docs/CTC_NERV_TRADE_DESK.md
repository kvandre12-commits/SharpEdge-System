# CTC/NERV Trade Desk

The CTC/NERV Trade Desk is a research-only workflow that connects:

1. **Institutional Regime Observatory / IRRP** — doctrine and evidence rules.
2. **CTC catalyst workbook** — regime-specific ticker universe, gates, falsifiers,
   provisional structures, and execution checks.
3. **NERV** — free/public option-chain discovery, liquidity scoring, and manual
   validation queue.
4. **Broker fresh-quote gate** — final bid/ask, size, buying-power impact, and
   order submission remain blocked until explicit operator approval.

This is a trade **desk**, not an auto-trader. Cute distinction. Very expensive if
ignored.

The CTC workbook is the first cartridge. The same taxonomy can be reused for
other company/institution workbooks with:

```bash
python3 scripts/regime_nerv_trade_desk.py --workbook path/to/company_pack.xlsx
```

See `docs/REGIME_CARTRIDGE_TAXONOMY.md` for the generic workbook contract.

## Build the board

Run NERV first if the liquidity board is stale or missing:

```bash
python3 scripts/nerv_free_data_adapter.py --symbols STNG,FRO,DHT,TNK,INSW,LPG,LNG,BKR,SLB,HAL,VLO,OKE --max-expirations 2 --board-limit 0
```

Then build the CTC/NERV desk board:

```bash
python3 scripts/ctc_nerv_trade_desk.py \
  --ctc-workbook '/sdcard/Download/CTC_C001_Full_34_Name_Disposition_v0_5 (1).xlsx'
```

Default inputs:

- CTC workbook: `/sdcard/Download/CTC_C001_Full_34_Name_Disposition_v0_5 (1).xlsx`
- NERV board: `outputs/nerv/nerv_liquidity_board.json`

Use `--board-limit 0` for desk runs so every CTC ticker can be matched. The default top-50 NERV board is useful for quick triage, but it can hide lower-ranked underlyings from the full trade-desk join.

Outputs:

- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.json`
- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.csv`
- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.md`

Generated `outputs/nerv/` and `outputs/nerv_trade_desk/` files are disposable runtime artifacts and are git-ignored. Both CLIs opportunistically prune stale files older than 24 hours before writing; pass `--retention-hours 0` to disable that broom for a forensic/debug run.

## Desk states

| State | Meaning |
|---|---|
| `needs_nerv_snapshot` | CTC has the ticker, but no NERV liquidity row exists yet. |
| `reject_options_for_now` | NERV found fatal quote/liquidity issues. |
| `refresh_quote_required` | Candidate exists, but fresh quote is missing or stale. |
| `manual_validate_candidate` | CTC + NERV are both clean enough for human structure review. |
| `equity_or_research_only` | CTC row is not currently options-led. |

## Non-negotiable execution rule

Every row remains execution-blocked until:

- broker fresh quote is observed;
- final debit/credit and size are confirmed;
- buying-power impact is confirmed;
- assignment/dividend hazards are checked;
- operator approval exists;
- non-SPY risk policy is explicitly expanded if trading non-SPY CTC names.

NERV public/free data may advance research. It does not authorize order entry.
