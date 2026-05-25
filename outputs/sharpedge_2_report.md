# SharpEdge 2.0 Auction Expectancy Report

- Generated UTC: 2026-05-25T23:41:06+00:00
- Symbol: SPY
- Minimum supported sample: 20

## Today’s Market State
- Date: 2026-05-20
- Regime: mrll0 / mid_vol|rising_voltrend|low_dp|low_macro|0_comp
- Vol state: mid / trend: rising
- Macro state: low
- Dark pool state: low
- Open resolution: NO_SETUP confidence=10.00
- Setup direction: NA
- Dealer hint: DEFENSIVE
- Spot: 742.72 | ATM: 743.00
- Gamma / OI wall: 500.00
- Call wall: 745.00 | Put wall: 535.00
- PCR OI: 2.33
- Signal bucket: quiet score=24.32
- Trade permission: 0

## Is The Current Gap-Fill Setup Tradable?
- Recommendation: **NORMAL**
- Reason: Supported edge, but still requires execution discipline.
- Squeeze risk: LOW
- Continuation risk: HIGH
- Best matching playbook condition: UNCLASSIFIED

- Sample support: SUPPORTED / n=28
- Expected fill probability: 78.6%
- Direct fill rate: 0.0%
- Failed fill rate: 0.0%
- Expected time-to-fill: 101.59 minutes
- Median time-to-fill: 97.50 minutes
- Expected MAE before fill: 0.5%
- Expected MFE: 0.4%
- Payoff ratio: NA
- Expectancy: 1.3813
- Sortino: NA
- Max drawdown: 0.0%
- Tradability score: 74.85

## Most Similar Supported Historical Paths
1. match=2 | n=28 | event=UNCLASSIFIED | path=SQUEEZE_THEN_FILL | regime=NA | open=NO_SETUP | fill=78.6% | expectancy=1.3813 | score=74.85
2. match=1 | n=29 | event=UNCLASSIFIED | path=UNCLASSIFIED | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00
3. match=0 | n=32 | event=UNCLASSIFIED | path=UNCLASSIFIED | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00

## Low-Sample Rows To Watch But Not Trust
1. LOW_SAMPLE n=8 | event=UNCLASSIFIED | path=ROTATIONAL_BALANCE_THEN_FILL | fill=100.0% | expectancy=246.5959 | score=9888.83
2. LOW_SAMPLE n=1 | event=CLEAN_BREAKOUT | path=UNCLASSIFIED | fill=100.0% | expectancy=54.4374 | score=2202.50
3. LOW_SAMPLE n=1 | event=CLEAN_BREAKDOWN | path=UNCLASSIFIED | fill=0.0% | expectancy=37.1546 | score=1486.18
4. LOW_SAMPLE n=2 | event=FAILED_BREAKDOWN | path=DIRECT_FILL | fill=100.0% | expectancy=15.7125 | score=653.49
5. LOW_SAMPLE n=5 | event=CLEAN_BREAKDOWN | path=SQUEEZE_THEN_FILL | fill=100.0% | expectancy=11.5757 | score=488.01
