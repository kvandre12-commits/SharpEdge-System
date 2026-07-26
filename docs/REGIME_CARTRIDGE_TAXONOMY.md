# Regime Cartridge Taxonomy

This is the generic workbook contract behind the CTC/NERV Trade Desk.

A **regime cartridge** is a company, institution, sector, or catalyst workbook
that turns a macro/institutional thesis into a ticker universe and a validation
queue. CTC-C001 is the first concrete cartridge, but the taxonomy is reusable.

## Pipeline

```text
Company / institution / catalyst
→ revealed-prior thesis
→ evidence class
→ catalyst gates and falsifiers
→ ticker universe
→ preferred vehicle and provisional structure
→ NERV liquidity validation
→ broker fresh-quote gate
```

## Generic command

```bash
python3 scripts/regime_nerv_trade_desk.py --workbook path/to/company_pack.xlsx
```

Equivalent legacy/CTC command:

```bash
python3 scripts/ctc_nerv_trade_desk.py --workbook path/to/company_pack.xlsx
```

Both commands use the same engine and write the same research-only artifacts under
`outputs/nerv_trade_desk/`.

## Required workbook sheets

The parser is intentionally tolerant about extra rows and layout text. It looks
for header rows in the first dozen rows of each sheet.

### `Full_Disposition_34` or `Company_Universe`

At least one of these sheets should exist.

Recommended fields:

| Field | Meaning |
|---|---|
| `Rank` | Cartridge rank / priority. |
| `Ticker` | Tradable underlying symbol. Required. |
| `Company` | Company or instrument name. |
| `Sleeve` / `Category` | Mechanism bucket. |
| `Primary_Mechanism` | Why this name transmits the regime. |
| `Current_Research_State` | Current thesis/readiness state. |
| `Preferred_Vehicle` | `OPTIONS`, `EQUITY`, `BASKET`, `NO_TRADE`, etc. |
| `Disposition_Score_0_100` | Cartridge-level priority score. |
| `Preferred_Structure` | Human-readable preferred option/equity structure. |
| `Thesis_Score` | Optional extra conviction score. |

### `I4_Playbook`, `Trade_Triage`, or `Options_Candidates`

Optional but recommended for options-led cartridges.

Recognized fields:

| Field | Meaning |
|---|---|
| `Ticker` | Underlying symbol. Required for row join. |
| `Primary_Structure` / `Structure` / `Structure_Category` | Proposed vehicle geometry. |
| `Expiry` / `Expiration` | Proposed expiry. |
| `DTE` | Days to expiry. |
| `Proxy_Debit_USD` / `Net_Debit_Proxy_USD` | Delayed/EOD debit estimate. |
| `Max_Loss_per_Spread_USD` | Optional risk proxy. |

### `I4_Execution_Check`

Optional execution gate sheet.

Recognized fields:

| Field | Meaning |
|---|---|
| `Ticker` | Underlying symbol. |
| `Fresh_Quote` | `YES` only when manually refreshed at broker/Cboe. Anything else blocks. |

## Output states

| State | Meaning |
|---|---|
| `needs_nerv_snapshot` | Workbook has ticker, but current NERV board has no row. |
| `reject_options_for_now` | NERV found fatal quote/liquidity issues. |
| `refresh_quote_required` | NERV found a candidate, but fresh quote is missing/stale. |
| `manual_validate_candidate` | Workbook + NERV are clean enough for human review. |
| `equity_or_research_only` | No options-led vehicle/structure is present. |

## Execution boundary

A regime cartridge never grants order authority.

Every row remains blocked until fresh broker quote, final debit/credit, size,
buying-power effect, assignment/dividend status, operator approval, and any
non-SPY risk-policy expansion are complete.
