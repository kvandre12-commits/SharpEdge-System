# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 506
- Baseline expectancy (all days): 0.000760
- Baseline sharpe_ann (all days): 1.170
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 255
- UNRESOLVED_PRESSURE: 146
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        506         255    0.503953       255  0.650980    0.002281    4.919457 4.948653 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001521       3.749709              True    True
        506         360    0.711462       360  0.644444    0.001914    4.750714 5.678189 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001154       3.580966              True    True
        506         360    0.711462       360  0.644444    0.001914    4.750714 5.678189 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001154       3.580966              True    True
        506         105    0.207510       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000262       4.665397              True    True
        506         506    1.000000       506  0.577075    0.000760    1.169748 1.657553 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        506         506    1.000000       506  0.577075    0.000760    1.169748 1.657553 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        506           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        506           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        506           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False