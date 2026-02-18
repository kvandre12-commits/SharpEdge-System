# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 512
- Baseline expectancy (all days): 0.000729
- Baseline sharpe_ann (all days): 1.126
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 260
- UNRESOLVED_PRESSURE: 147
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        512         260    0.507812       260  0.650000    0.002254    4.900823 4.978006 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001524       3.774785              True    True
        512         365    0.712891       365  0.643836    0.001900    4.741713 5.706652 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001170       3.615674              True    True
        512         365    0.712891       365  0.643836    0.001900    4.741713 5.706652 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001170       3.615674              True    True
        512         105    0.205078       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000293       4.709106              True    True
        512         512    1.000000       512  0.576172    0.000729    1.126038 1.605048 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        512         512    1.000000       512  0.576172    0.000729    1.126038 1.605048 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        512           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        512           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        512           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False