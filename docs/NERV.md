# NERV: Normalized Edge Research & Validation

NERV is the free-data options research architecture for building systematic option-chain, surface, liquidity, and structure analysis without pretending that free public data is execution-grade OPRA NBBO.

It is a **SharpEdge research and validation pipeline**, not an autonomous execution system.

## Core Principle

Public/free data may authorize research advancement. It may not authorize final order entry.

The execution broker remains the only acceptable source for:

- final bid/ask;
- current complex-order market;
- available size;
- buying-power effect;
- assignment and dividend status;
- final maximum debit/credit;
- order submission.

## Recommended Free Stack

### 1. Tradier Brokerage API

Tradier is the strongest free-or-zero-incremental-cost programmatic route when opening or maintaining an account is acceptable.

Expected brokerage API strengths:

- real-time U.S. equity and options market data;
- full option-chain access;
- option expirations, quotes, chains, and pricing;
- Greeks updated hourly;
- suitable as the principal research feed.

Sandbox limitations:

- chain data delayed approximately 15 minutes;
- no supplied Greeks;
- strikes, expirations, bid, ask, last, volume, and open interest available;
- Greeks must be calculated internally by NERV.

Use Tradier as the preferred source if account access and API credentials are available.

### 2. Alpaca Basic

Alpaca Basic is the cleanest explicitly free authenticated API route if Tradier is not used.

Strengths:

- all U.S. option securities;
- indicative options pricing feed;
- option-chain endpoints;
- 200 historical API calls per minute;
- 200 option quote websocket subscriptions;
- historical data excluding the most recent 15 minutes.

Limitations:

- Basic uses an indicative derivative of OPRA, not consolidated OPRA NBBO;
- appropriate for screening, surface construction, and provisional structure analysis;
- not suitable for final execution pricing.

Alpaca may supply implied volatility and Greeks when:

- bid and ask are nonzero;
- the underlying has a valid trade;
- the option has not expired;
- the mathematical solution is valid.

It may omit Greeks for illiquid contracts and does not calculate them for 0DTE contracts.

### 3. yfinance

yfinance is the bulk discovery and fallback layer, not the source of record.

Useful fields:

- underlying history;
- expiration dates;
- call and put chains;
- contract symbols;
- strikes;
- last trade date;
- last price;
- bid and ask;
- volume;
- open interest;
- implied volatility;
- in-the-money state.

Use yfinance for:

- rapid multi-name chain enumeration;
- finding optionable names;
- identifying expirations;
- creating strike ladders;
- first-pass spread candidates;
- detecting obviously unusable chains.

Do not use yfinance as sole authority for:

- final bid/ask;
- timestamp precision;
- executable NBBO;
- unusual corporate-action options;
- thinly traded tanker and shipping names;
- 0DTE decisions.

### 4. OCC

OCC is the official contract and liquidity-validation layer.

Authoritative uses:

- listed series;
- expiration dates;
- strikes;
- call and put open interest;
- daily and historical volume;
- daily open-interest batch files.

OCC does not provide:

- bid;
- ask;
- implied volatility;
- delta;
- gamma;
- theta;
- vega.

NERV should use OCC underneath Tradier, Alpaca, or yfinance to validate contract existence, volume, and open interest.

### 5. Cboe Delayed Chains

Cboe public delayed chains are a high-quality manual validation source.

Available manually:

- delayed underlying bid, ask, and last;
- option last;
- bid and ask;
- volume;
- implied volatility;
- delta;
- gamma;
- open interest;
- multiple expirations;
- manual CSV download.

Important restriction:

- Cboe public delayed tables must not be used as an automated bulk-scraping backend.

Use Cboe for:

- manual validation of finalists;
- yfinance/Alpaca discrepancy checks;
- delta and gamma checks when automated sources lack them;
- bid/ask quality checks;
- selected manual workbook exports.

## Manual Fallback Sources

### Nasdaq

Use as occasional manual corroboration only. Public pages are delayed and are not a documented bulk options API.

### Barchart

Use as a manual delayed screener for:

- liquidity;
- unusual volume;
- put/call activity;
- broad candidate discovery;
- active-chain verification.

Do not treat Barchart probability calculations as independent evidence.

### MarketBeat

Use as an end-of-day fallback and historical record, not final bid/ask.

### OptionStrat

Use for manual structure visualization:

- payoff geometry;
- breakevens;
- profit zones;
- calendar/diagonal behavior;
- IV-change sensitivity.

### AlphaQuery

Use for IV-regime and term-context cross-checks:

- mean IV by horizon;
- call IV;
- put IV;
- put/call IV ratios;
- historical volatility charts.

## Source Hierarchy

1. Tradier brokerage API — strongest if account access is acceptable.
2. Alpaca Basic — strongest explicitly free API.
3. yfinance — bulk chain discovery and fallback.
4. OCC — official series, volume, and open-interest validation.
5. Cboe — manual high-quality delayed-chain validation.
6. Barchart / MarketBeat — delayed screening and EOD fallback.
7. OptionStrat / AlphaQuery — payoff and IV-regime validation.
8. Nasdaq — occasional manual corroboration.

## Automated Nightly Population

Primary programmatic source:

- Alpaca Basic API; or
- Tradier brokerage API if account access is established.

Secondary source:

- yfinance bulk chain discovery.

Official validation:

- OCC daily open-interest CSV;
- OCC volume and series data.

For each ticker and expiration, ingest:

- underlying price and timestamp;
- contract symbol;
- call/put;
- expiration;
- strike;
- bid;
- ask;
- last;
- volume;
- open interest;
- supplied implied volatility;
- supplied Greeks where available;
- source;
- data mode;
- quote age.

## Internally Calculated Fields

Where source Greeks or derived fields are unavailable, NERV calculates:

- midpoint;
- bid/ask width;
- implied volatility;
- delta;
- gamma;
- theta;
- vega;
- intrinsic value;
- extrinsic value;
- moneyness;
- expected move;
- call/put skew;
- expiration term structure;
- dollars per delta;
- spread net Greeks;
- maximum gain and loss;
- breakeven;
- expected-move coverage.

For dividend-paying American equity options, internally calculated Greeks must be flagged as model-derived estimates, especially for:

- deep-in-the-money puts;
- contracts near ex-dividend dates.

## Manual Validation Queue

Validate only candidates that survive automated screening.

Priority names:

1. STNG
2. FRO
3. DHT
4. TNK
5. INSW
6. LPG
7. LNG
8. BKR
9. SLB
10. HAL
11. VLO
12. OKE

## SharpEdge Integration

NERV belongs in SharpEdge's analytics/research layer. It should feed research tables, surface snapshots, structure candidates, and workbook outputs. It must not bypass the operator approval model or broker confirmation gate.

Natural integration points:

- `scripts/` for standalone ingestion, normalization, and surface-building CLIs;
- `data/*.db` for persistent normalized chain, surface, and validation tables;
- `outputs/` for candidate boards, workbook exports, and manual-validation queues;
- operator artifacts only after candidates pass research scoring and quote-quality gates.

## Current Implementation

NERV-1 has an initial SharpEdge implementation:

```bash
python3 scripts/nerv_free_data_adapter.py --symbols STNG,FRO,DHT --max-expirations 2
python3 scripts/nerv_free_data_adapter.py --status-only
```

Current artifacts:

- `outputs/nerv/nerv_provider_status.json`;
- `outputs/nerv/nerv_options_snapshot.json`;
- `outputs/nerv/nerv_options_snapshot.csv`;
- `outputs/nerv/nerv_liquidity_board.json`;
- `outputs/nerv/nerv_liquidity_board.csv`.

Implemented now:

- yfinance bulk discovery adapter;
- OCC compact symbol parser/formatter;
- provider credential status for Tradier, Alpaca, and yfinance;
- normalized source, data mode, fetch timestamp, quote timestamp, and quote age;
- midpoint, bid/ask width, and moneyness derived fields;
- quote-quality, liquidity, and combined NERV scores;
- manual-validation priority labels;
- rejection flags for stale, missing, crossed, tiny, or inactive markets;
- sorted liquidity board for finalist triage;
- research-only warning on every exported row.

Still pending:

- Tradier live/sandbox adapter;
- Alpaca Basic official adapter;
- OCC daily open-interest batch loader;
- SQLite persistence for NERV-specific multi-name chain snapshots;
- surface compiler and structure generator;
- tuned scoring thresholds by strategy family and DTE bucket.

Validation performed:

- unit tests for OCC symbols, derived quote fields, yfinance normalization, scorer ranking, and scored writers;
- compile check for NERV modules and CLI;
- one-symbol SPY smoke fetch produced 272 normalized quotes from one expiration.

## CTC/NERV Trade Desk

NERV now has a CTC trade-desk bridge for catalyst-cartridge workflows.

```bash
python3 scripts/nerv_free_data_adapter.py --symbols STNG,FRO,DHT,TNK,INSW,LPG,LNG,BKR,SLB,HAL,VLO,OKE --max-expirations 2 --board-limit 0
python3 scripts/ctc_nerv_trade_desk.py \
  --ctc-workbook '/sdcard/Download/CTC_C001_Full_34_Name_Disposition_v0_5 (1).xlsx'
```

Use `--board-limit 0` for full CTC desk runs so the NERV liquidity board includes every scored contract rather than only the top triage rows.

The bridge joins CTC disposition rows, CTC provisional structures, CTC execution-check rows, and the current NERV liquidity board into:

- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.json`;
- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.csv`;
- `outputs/nerv_trade_desk/ctc_nerv_trade_desk.md`.

Desk output is research-only. It may promote a name to manual validation, but every row remains blocked until broker fresh quote, final debit/credit, buying-power impact, assignment/dividend checks, explicit operator approval, and any required non-SPY risk-policy expansion are complete.

See `docs/CTC_NERV_TRADE_DESK.md`.

## NERV Build Plan

### NERV-1: Free Data Adapter

- Tradier authentication, if available;
- Alpaca Basic authentication, if available;
- yfinance backup adapter;
- OCC batch loader;
- normalized OCC contract symbols;
- timestamp logging;
- source and data-mode logging.

### NERV-2: Surface Compiler

- delta ladder;
- strike ladder;
- term structure;
- skew;
- expected move;
- liquidity scores;
- structure generator.

### NERV-3: Manual Validation

- Cboe or broker snapshots for finalists;
- discrepancy checks;
- quote-quality grading;
- delayed/manual-source provenance.

### NERV-4: Workbook Population

- populate all options-led names;
- run rejection tests for equity-first names;
- generate alternate structures;
- enforce fresh-quote execution gate.

## Practical Determination

NERV should use Alpaca Basic or Tradier as the programmatic foundation, yfinance as the broad discovery layer, OCC as the official contract/liquidity authority, and Cboe or the execution broker as the manual/final validation layer.

This fills nearly every analytical field for free.

What it cannot provide for free and unattended is consistently reliable, execution-grade OPRA NBBO across the full universe at the instant of entry.
