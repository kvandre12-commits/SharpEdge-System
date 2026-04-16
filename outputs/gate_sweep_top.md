# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 552
- Baseline expectancy (all days): 0.000725
- Baseline sharpe_ann (all days): 1.119
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 289
- UNRESOLVED_PRESSURE: 155
- COILED: 108

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        552         289    0.523551       289  0.633218    0.001947    4.130466 4.423313 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001221       3.011643              True    True
        552         397    0.719203       397  0.632242    0.001718    4.156238 5.216694 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000992       3.037415              True    True
        552         397    0.719203       397  0.632242    0.001718    4.156238 5.216694 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000992       3.037415              True    True
        552         108    0.195652       108  0.629630    0.001105    6.158559 4.031723 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000380       5.039736              True    True
        552         552    1.000000       552  0.572464    0.000725    1.118823 1.655887 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        552         552    1.000000       552  0.572464    0.000725    1.118823 1.655887 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        552           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        552           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        552           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False