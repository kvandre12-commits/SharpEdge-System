# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 575
- Baseline expectancy (all days): 0.000791
- Baseline sharpe_ann (all days): 1.234
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 296
- UNRESOLVED_PRESSURE: 163
- COILED: 116

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        575         296    0.514783       296  0.631757    0.001999    4.262702 4.619878 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001208       3.028435              True    True
        575         412    0.716522       412  0.631068    0.001750    4.272179 5.462580 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000959       3.037912              True    True
        575         412    0.716522       412  0.631068    0.001750    4.272179 5.462580 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000959       3.037912              True    True
        575         116    0.201739       116  0.629310    0.001115    6.106815 4.143272 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000324       4.872548              True    True
        575         575    1.000000       575  0.572174    0.000791    1.234267 1.864416 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        575         575    1.000000       575  0.572174    0.000791    1.234267 1.864416 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        575           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        575           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        575           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False