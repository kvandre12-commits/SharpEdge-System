# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 543
- Baseline expectancy (all days): 0.000615
- Baseline sharpe_ann (all days): 0.948
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 284
- UNRESOLVED_PRESSURE: 154
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        543         284     0.52302       284  0.626761    0.001879    3.974185 4.218975 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001264       3.025946              True    True
        543         389     0.71639       389  0.627249    0.001648    3.975081 4.938789 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001033       3.026841              True    True
        543         389     0.71639       389  0.627249    0.001648    3.975081 4.938789 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001033       3.026841              True    True
        543         105     0.19337       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000408       4.886906              True    True
        543         543     1.00000       543  0.567219    0.000615    0.948239 1.391931 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        543         543     1.00000       543  0.567219    0.000615    0.948239 1.391931 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        543           0     0.00000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        543           0     0.00000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        543           0     0.00000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False