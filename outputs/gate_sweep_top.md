# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 547
- Baseline expectancy (all days): 0.000668
- Baseline sharpe_ann (all days): 1.028
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 286
- UNRESOLVED_PRESSURE: 155
- COILED: 106

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        547         286    0.522852       286  0.629371    0.001870    3.969890 4.229230 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001202       2.941522              True    True
        547         392    0.716636       392  0.630102    0.001650    3.996000 4.983888 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000982       2.967632              True    True
        547         392    0.716636       392  0.630102    0.001650    3.996000 4.983888 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000982       2.967632              True    True
        547         106    0.193784       106  0.632075    0.001058    6.012936 3.899773 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000390       4.984568              True    True
        547         547    1.000000       547  0.570384    0.000668    1.028369 1.515103 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        547         547    1.000000       547  0.570384    0.000668    1.028369 1.515103 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        547           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        547           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        547           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False