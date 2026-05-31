# SharpEdge 2.0 Auction Expectancy Report

- Generated UTC: 2026-05-31T23:55:42+00:00
- Symbol: SPY
- Minimum supported sample: 20

## Today’s Market State
- Date: 2026-05-20
- Regime: mrll0 / mid_vol|rising_voltrend|low_dp|low_macro|0_comp
- Vol state: mid / trend: rising
- Macro state: low
- Dark pool state: low
- Open resolution: NO_SETUP confidence=10.00
- Setup direction: NONE
- Dealer hint: DEFENSIVE
- Spot: 742.72 | ATM: 743.00
- Gamma / OI wall: 500.00
- Call wall: 745.00 | Put wall: 535.00
- PCR OI: 2.33
- Signal bucket: quiet score=24.32
- Trade permission: 0

## Is The Current Gap-Fill Setup Tradable?
- Recommendation: **DO_NOTHING**
- Reason: No positive supported edge.
- Squeeze risk: LOW
- Continuation risk: LOW
- Best matching playbook condition: UNCLASSIFIED

- Sample support: SUPPORTED / n=27
- Expected fill probability: 0.0%
- Direct fill rate: 0.0%
- Failed fill rate: 0.0%
- Expected time-to-fill: NA minutes
- Median time-to-fill: NA minutes
- Expected MAE before fill: NA
- Expected MFE: NA
- Payoff ratio: NA
- Expectancy: 0.0000
- Sortino: NA
- Max drawdown: 0.0%
- Tradability score: 0.00

## Most Similar Supported Historical Paths
1. match=2 | n=27 | event=UNCLASSIFIED | path=UNCLASSIFIED | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00
2. match=1 | n=31 | event=UNCLASSIFIED | path=UNCLASSIFIED | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00

## Low-Sample Rows To Watch But Not Trust
1. LOW_SAMPLE n=4 | event=CLEAN_BREAKDOWN | path=SQUEEZE_THEN_FILL | fill=100.0% | expectancy=10.0807 | score=428.20
2. LOW_SAMPLE n=4 | event=UNCLASSIFIED | path=DIRECT_FILL | fill=100.0% | expectancy=4.5644 | score=207.56
3. LOW_SAMPLE n=1 | event=FAILED_BREAKDOWN | path=SQUEEZE_THEN_FILL | fill=100.0% | expectancy=3.3980 | score=160.86
4. LOW_SAMPLE n=2 | event=UNCLASSIFIED | path=DIRECT_FILL | fill=100.0% | expectancy=2.0823 | score=108.27
5. LOW_SAMPLE n=6 | event=UNCLASSIFIED | path=SQUEEZE_THEN_FILL | fill=100.0% | expectancy=1.8342 | score=98.32
