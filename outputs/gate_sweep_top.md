# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 530
- Baseline expectancy (all days): 0.000648
- Baseline sharpe_ann (all days): 1.008
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 274
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        530         274    0.516981       274  0.635036    0.002064    4.460047 4.650659 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001416       3.452437              True    True
        530         379    0.715094       379  0.633245    0.001776    4.384345 5.376802 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001128       3.376735              True    True
        530         379    0.715094       379  0.633245    0.001776    4.384345 5.376802 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001128       3.376735              True    True
        530         105    0.198113       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000375       4.827535              True    True
        530         530    1.000000       530  0.569811    0.000648    1.007610 1.461269 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        530         530    1.000000       530  0.569811    0.000648    1.007610 1.461269 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        530           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        530           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        530           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False