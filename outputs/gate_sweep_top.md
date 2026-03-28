# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 540
- Baseline expectancy (all days): 0.000557
- Baseline sharpe_ann (all days): 0.863
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 282
- UNRESOLVED_PRESSURE: 153
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        540         282    0.522222       282  0.627660    0.001877    3.964253 4.193586 -0.079438                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001320       3.101521              True    True
        540         387    0.716667       387  0.627907    0.001645    3.966291 4.915184 -0.079438                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001089       3.103560              True    True
        540         387    0.716667       387  0.627907    0.001645    3.966291 4.915184 -0.079438                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001089       3.103560              True    True
        540         105    0.194444       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000466       4.972414              True    True
        540         540    1.000000       540  0.566667    0.000557    0.862731 1.262909 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        540         540    1.000000       540  0.566667    0.000557    0.862731 1.262909 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        540           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        540           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        540           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False