# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 577
- Baseline expectancy (all days): 0.000794
- Baseline sharpe_ann (all days): 1.240
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 298
- UNRESOLVED_PRESSURE: 163
- COILED: 116

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        577         298    0.516464       298  0.630872    0.001997    4.255604 4.627742 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001203       3.015265              True    True
        577         414    0.717504       414  0.630435    0.001750    4.265570 5.467351 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000956       3.025230              True    True
        577         414    0.717504       414  0.630435    0.001750    4.265570 5.467351 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000956       3.025230              True    True
        577         116    0.201040       116  0.629310    0.001115    6.106815 4.143272 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000320       4.866475              True    True
        577         577    1.000000       577  0.571924    0.000794    1.240340 1.876844 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        577         577    1.000000       577  0.571924    0.000794    1.240340 1.876844 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        577           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False