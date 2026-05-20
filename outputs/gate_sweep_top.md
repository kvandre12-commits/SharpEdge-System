# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 577
- Baseline expectancy (all days): 0.000788
- Baseline sharpe_ann (all days): 1.232
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 297
- UNRESOLVED_PRESSURE: 163
- COILED: 117

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        577         297    0.514731       297  0.629630    0.001970    4.198044 4.557481 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001181       2.966206              True    True
        577         414    0.717504       414  0.630435    0.001742    4.251446 5.449247 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000953       3.019608              True    True
        577         414    0.717504       414  0.630435    0.001742    4.251446 5.449247 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000953       3.019608              True    True
        577         117    0.202773       117  0.632479    0.001164    6.298165 4.291476 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000376       5.066328              True    True
        577         577    1.000000       577  0.571924    0.000788    1.231837 1.863979 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        577         577    1.000000       577  0.571924    0.000788    1.231837 1.863979 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False