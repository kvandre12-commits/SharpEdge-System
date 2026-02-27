# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 519
- Baseline expectancy (all days): 0.000738
- Baseline sharpe_ann (all days): 1.144
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 266
- UNRESOLVED_PRESSURE: 148
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        519         266    0.512524       266  0.650376    0.002277    4.976578 5.112948 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001539       3.832998              True    True
        519         371    0.714836       371  0.644205    0.001922    4.807280 5.832921 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001184       3.663700              True    True
        519         371    0.714836       371  0.644205    0.001922    4.807280 5.832921 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001184       3.663700              True    True
        519         105    0.202312       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000285       4.691565              True    True
        519         519    1.000000       519  0.576108    0.000738    1.143580 1.641157 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        519         519    1.000000       519  0.576108    0.000738    1.143580 1.641157 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        519           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        519           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        519           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False