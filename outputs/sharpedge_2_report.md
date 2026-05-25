# SharpEdge 2.0 Auction Expectancy Report

- Generated UTC: 2026-05-25T22:19:00+00:00
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
- Continuation risk: MEDIUM
- Best matching playbook condition: UNCLASSIFIED

- Sample support: SUPPORTED / n=64
- Expected fill probability: 60.9%
- Direct fill rate: 0.0%
- Failed fill rate: 0.0%
- Expected time-to-fill: 95.38 minutes
- Median time-to-fill: 90.00 minutes
- Expected MAE before fill: 0.4%
- Expected MFE: 0.4%
- Payoff ratio: NA
- Expectancy: 32.7249
- Sortino: NA
- Max drawdown: 0.0%
- Tradability score: 1324.19

## Most Similar Supported Historical Paths
1. match=2 | n=64 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=60.9% | expectancy=32.7249 | score=1324.19
2. match=2 | n=49 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=75.5% | expectancy=5.0794 | score=222.02
3. match=2 | n=24 | event=CLEAN_BREAKDOWN | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=16.7% | expectancy=2.7643 | score=114.65
4. match=2 | n=25 | event=CLEAN_BREAKOUT | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=8.0% | expectancy=0.1666 | score=8.55
5. match=1 | n=29 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=41.4% | expectancy=71.4889 | score=332539.06
6. match=1 | n=22 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=72.7% | expectancy=19.1611 | score=172622.97
7. match=1 | n=33 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=NO_SETUP | fill=69.7% | expectancy=21.2939 | score=869.15
8. match=1 | n=29 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00
9. match=0 | n=32 | event=UNCLASSIFIED | path=UNKNOWN | regime=NA | open=UNKNOWN | fill=0.0% | expectancy=0.0000 | score=0.00

## Low-Sample Rows To Watch But Not Trust
1. LOW_SAMPLE n=4 | event=CLEAN_BREAKOUT | path=UNKNOWN | fill=100.0% | expectancy=14.9324 | score=622.26
2. LOW_SAMPLE n=9 | event=FAILED_BREAKDOWN | path=UNKNOWN | fill=100.0% | expectancy=7.5647 | score=327.53
3. LOW_SAMPLE n=12 | event=CLEAN_BREAKDOWN | path=UNKNOWN | fill=100.0% | expectancy=7.5074 | score=325.28
4. LOW_SAMPLE n=5 | event=FAILED_BREAKDOWN | path=UNKNOWN | fill=100.0% | expectancy=5.1597 | score=231.33
5. LOW_SAMPLE n=5 | event=FAILED_BREAKOUT | path=UNKNOWN | fill=100.0% | expectancy=4.9999 | score=224.95
