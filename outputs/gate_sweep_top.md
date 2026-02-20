# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 514
- Baseline expectancy (all days): 0.000731
- Baseline sharpe_ann (all days): 1.131
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 262
- UNRESOLVED_PRESSURE: 147
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        514         262    0.509728       262  0.648855    0.002246    4.896541 4.992749 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001515       3.765780              True    True
        514         367    0.714008       367  0.643052    0.001896    4.740285 5.720542 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001165       3.609524              True    True
        514         367    0.714008       367  0.643052    0.001896    4.740285 5.720542 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001165       3.609524              True    True
        514         105    0.204280       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000291       4.704384              True    True
        514         514    1.000000       514  0.575875    0.000731    1.130761 1.614924 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        514         514    1.000000       514  0.575875    0.000731    1.130761 1.614924 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        514           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        514           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        514           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False