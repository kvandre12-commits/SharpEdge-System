# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 539
- Baseline expectancy (all days): 0.000589
- Baseline sharpe_ann (all days): 0.915
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 281
- UNRESOLVED_PRESSURE: 153
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        539         281    0.521336       281  0.629893    0.001944    4.146429 4.378518 -0.063468                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001355       3.231393              True    True
        539         386    0.716141       386  0.629534    0.001694    4.121149 5.100488 -0.063468                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001104       3.206113              True    True
        539         386    0.716141       386  0.629534    0.001694    4.121149 5.100488 -0.063468                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001104       3.206113              True    True
        539         105    0.194805       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000433       4.920109              True    True
        539         539    1.000000       539  0.567718    0.000589    0.915036 1.338235 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        539         539    1.000000       539  0.567718    0.000589    0.915036 1.338235 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        539           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        539           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        539           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False