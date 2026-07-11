# SE short detector

## Purpose

The **SE short detector** is a support-layer scanner for catching those little
"what the hell is this" tickers that suddenly print absurd volume and start
behaving like low-quality squeeze / promo / mania names.

This is **not** an execution engine and **not** an autonomous shorting system.
It is a curiosity-to-operator detector:

- find weird names
- rank how abnormal they are
- explain why they are abnormal
- let the operator decide whether the name is interesting, dangerous, or short-worthy

## Naming doctrine

To avoid confusion, use one name consistently:

- **SE short detector**

Avoid near-duplicate labels like:

- microcap scanner
- meme detector
- junk ticker detector
- weird volume scanner

Those can appear in prose as descriptions, but the system name should stay
**SE short detector**.

## Core question

The detector is trying to answer:

> "Which small trashy-looking tickers are trading such abnormal volume and range
> that I need to stop and look at them right now?"

That is slightly different from:

- "Which names are good longs?"
- "Which names should be auto-shorted?"
- "Which names merely have high absolute volume?"

We care about **abnormal participation + suspicious price behavior**.

## Operator use case

The operator sees a ranked list like:

```text
1. XYZW — SE short detector score 91
   price 3.82 | rvol 14.7x | dollar vol 62x baseline | range 28% | float turnover high
   reason: tiny name, huge participation spike, vertical move, blow-off shape

2. QNTR — SE short detector score 84
   price 7.10 | rvol 9.4x | gap 22% | halt/news flag | weak liquidity quality
   reason: abnormal open expansion, unstable tape, suspect continuation odds
```

The output should feel like a **watchlist + reason card**, not a black box.

## V1 design goal

Build the smallest useful version first.

V1 should answer:

- is this ticker unusually active relative to itself?
- is the move unusually stretched?
- is the name small / low-quality enough to belong in the detector universe?
- is the tape acting like an ignition / squeeze / blow-off candidate?

Do **not** begin with:

- perfect float data dependency
- expensive full-market institutional feeds
- social sentiment overengineering
- auto-execution logic

Classic YAGNI. No need to bolt a jet engine onto a bicycle.

## Detection shape

The SE short detector can be thought of in four stages.

### 1. Universe selection

Start with names likely to produce the behavior we care about.

Possible entry rules:

- price between roughly **$1 and $20**
- exclude mega caps / broad ETFs
- include only names with live session trading activity
- optionally focus on top percentage gainers, top volume gainers, or top unusual volume lists

Possible later enrichments:

- market cap cutoff
- float cutoff
- known low-float flag
- recent reverse split flag
- offering / dilution history flag

## 2. Abnormal participation filters

These decide whether the ticker is even weird enough to care about.

Primary signals:

- **relative volume (RVOL)** = current session volume / typical session volume by this time of day
- **dollar volume expansion** = current dollar volume / typical dollar volume
- **float turnover proxy** = shares traded / float, if float is available
- **trade acceleration** = how quickly recent volume is increasing

Good v1 thresholds could look like:

- RVOL >= 5x
- dollar volume expansion >= 8x
- price change >= 12%
- intraday range >= 10%

Not exact doctrine yet — just practical starting rails.

## 3. Tape-behavior classification

Once a name is weird enough, classify what kind of weird it is.

Suggested SE short detector states:

- **IGNITION**
  - first abnormal expansion
  - volume and range both inflecting upward
  - often news / rumor / social catalyst zone

- **SQUEEZE CONTINUATION**
  - persistent upward pressure
  - elevated RVOL staying elevated
  - shallow pullbacks, trend holding

- **BLOW-OFF / EXHAUSTION**
  - extreme extension from VWAP or opening base
  - large upper wicks / failed continuation bars
  - momentum deceleration after vertical phase

- **CHAOTIC TRASH / UNSTABLE**
  - violent movement but poor continuity
  - repeated failed pushes
  - sloppy liquidity, spread problems, halt risk

This keeps the detector useful even when a name is **not** a short yet.
Sometimes the right answer is:

> "Yeah, weird as hell, but not ready." 

## 4. Scoring and explanation

Produce a single top-line score, but keep the reasons visible.

Example score components:

- abnormal volume score
- abnormal dollar-volume score
- range expansion score
- extension-from-VWAP score
- liquidity instability score
- blow-off / exhaustion score
- small-cap / low-quality universe fit score

Example output shape:

```json
{
  "symbol": "XYZW",
  "detector": "SE short detector",
  "score": 91,
  "state": "BLOW_OFF / EXHAUSTION",
  "metrics": {
    "price": 3.82,
    "rvol": 14.7,
    "dollar_volume_expansion": 62.0,
    "day_change_pct": 31.4,
    "intraday_range_pct": 28.2,
    "extension_from_vwap_pct": 9.1
  },
  "reasons": [
    "RVOL massively above baseline",
    "Dollar volume exploded versus normal participation",
    "Vertical extension now far from VWAP",
    "Tape shape suggests blow-off rather than healthy trend"
  ]
}
```

## Practical feature list

### V1 features

Cheap and realistic:

- symbol
- last price
- day % change
- session volume
- average historical volume
- relative volume
- dollar volume
- intraday high/low range %
- extension from VWAP
- recent momentum burst
- wickiness / rejection behavior

### V2 features

Useful, but not required for first pass:

- float
- float turnover
- shares outstanding
- market cap
- halt history intraday
- news presence / lack of credible catalyst
- offering / dilution / shelf / reverse split metadata
- options availability

### V3 features

Only if they clearly improve signal quality:

- promoter / social burst proxies
- borrow-fee / short-interest context
- premarket-to-open transition features
- multi-day mania lifecycle state

## Recommended V1 heuristics

A ticker becomes an **SE short detector candidate** when it passes something like:

- price in small-name band
- not ETF
- RVOL above threshold
- dollar volume expansion above threshold
- day change or range expansion above threshold

Then the score gets boosted by:

- extreme extension from VWAP
- acceleration after already-large move
- repeated upper rejection wicks
- failed breakout continuation after a vertical run
- unstable liquidity / gapy prints

And penalized by:

- strong orderly trend continuation with no exhaustion signs yet
- insufficient dollar liquidity
- stale move already dead and irrelevant

## Baseline math

The baseline matters a lot, otherwise everything becomes "wow volume" because a
stock printed one busy day.

Better baseline ideas:

- compare current cumulative volume to the **median** cumulative volume by the same time over the last 20 sessions
- compare dollar volume by time-of-day, not just end-of-day averages
- use medians instead of means when possible to reduce outlier distortion

That keeps the detector from getting fooled by one prior freak day.

## Output surfaces

The SE short detector should probably emit:

- a ranked terminal table for quick scan
- a JSON artifact for downstream use
- a human-readable reason card per symbol

Suggested artifact names:

- `outputs/se_short_detector.json`
- `outputs/se_short_detector.md`

## First implementation seam

Safest first seam:

- add a standalone script
- keep it outside the core cockpit decision engine
- generate a separate artifact
- do not touch SPY execution logic

Good starting shape:

```text
scripts/se_short_detector.py
outputs/se_short_detector.json
```

That keeps the experiment isolated and honest.

## Pseudocode sketch

```text
fetch candidate universe
for each symbol:
    fetch intraday price + volume context
    compute time-of-day relative volume
    compute dollar volume expansion
    compute intraday range and extension from VWAP
    compute rejection / exhaustion hints
    compute SE short detector score
    classify state
rank descending by score
write JSON + terminal summary
```

## What success looks like

The detector is doing its job if it makes the operator say:

- "Yep, that absolutely belongs on the weird board"
- "This one is still squeezing, leave it alone for now"
- "This one looks like late-stage trash exhaustion"
- "This is high volume, but not the kind of weird we care about"

That last one matters. The system should filter **interesting weird** from
plain busy names.

## Next step

If we want to move from sketch to implementation, the next clean step is:

1. define the data source for candidate symbols
2. define a minimal feature contract
3. build `scripts/se_short_detector.py`
4. emit a ranked artifact
5. validate manually on a few known garbage-rip days
