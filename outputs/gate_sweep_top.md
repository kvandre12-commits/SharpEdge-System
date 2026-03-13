# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 529
- Baseline expectancy (all days): 0.000660
- Baseline sharpe_ann (all days): 1.026
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 273
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        529         273    0.516068       273  0.637363    0.002093    4.522086 4.706737 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001433       3.496532              True    True
        529         378    0.714556       378  0.634921    0.001795    4.434921 5.431646 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001136       3.409366              True    True
        529         378    0.714556       378  0.634921    0.001795    4.434921 5.431646 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001136       3.409366              True    True
        529         105    0.198488       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000363       4.809590              True    True
        529         529    1.000000       529  0.570888    0.000660    1.025555 1.485889 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        529         529    1.000000       529  0.570888    0.000660    1.025555 1.485889 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        529           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        529           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        529           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False