# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 576
- Baseline expectancy (all days): 0.000778
- Baseline sharpe_ann (all days): 1.215
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 297
- UNRESOLVED_PRESSURE: 163
- COILED: 116

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        576         297    0.515625       297  0.629630    0.001970    4.198044 4.557481 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001192       2.983478              True    True
        576         413    0.717014       413  0.629540    0.001729    4.219031 5.401165 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000952       3.004466              True    True
        576         413    0.717014       413  0.629540    0.001729    4.219031 5.401165 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000952       3.004466              True    True
        576         116    0.201389       116  0.629310    0.001115    6.106815 4.143272 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000337       4.892249              True    True
        576         576    1.000000       576  0.571181    0.000778    1.214565 1.836250 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        576         576    1.000000       576  0.571181    0.000778    1.214565 1.836250 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        576           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        576           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        576           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False