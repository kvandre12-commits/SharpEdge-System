# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 544
- Baseline expectancy (all days): 0.000615
- Baseline sharpe_ann (all days): 0.950
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 285
- UNRESOLVED_PRESSURE: 154
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        544         285    0.523897       285  0.628070    0.001875    3.973808 4.225996 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001260       3.023885              True    True
        544         390    0.716912       390  0.628205    0.001646    3.975506 4.945662 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001030       3.025582              True    True
        544         390    0.716912       390  0.628205    0.001646    3.975506 4.945662 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001030       3.025582              True    True
        544         105    0.193015       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000407       4.885221              True    True
        544         544    1.000000       544  0.568015    0.000615    0.949924 1.395687 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        544         544    1.000000       544  0.568015    0.000615    0.949924 1.395687 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        544           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        544           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        544           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False