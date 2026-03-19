# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 533
- Baseline expectancy (all days): 0.000642
- Baseline sharpe_ann (all days): 0.999
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 277
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        533         277    0.519700       277  0.635379    0.002038    4.379275 4.591365 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001396       3.380638              True    True
        533         382    0.716698       382  0.633508    0.001759    4.315924 5.313800 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001117       3.317288              True    True
        533         382    0.716698       382  0.633508    0.001759    4.315924 5.313800 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001117       3.317288              True    True
        533         105    0.196998       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000380       4.836508              True    True
        533         533    1.000000       533  0.570356    0.000642    0.998636 1.452348 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        533         533    1.000000       533  0.570356    0.000642    0.998636 1.452348 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        533           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        533           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        533           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False