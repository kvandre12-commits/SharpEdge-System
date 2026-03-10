# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 526
- Baseline expectancy (all days): 0.000698
- Baseline sharpe_ann (all days): 1.084
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 270
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        526         270    0.513308       270  0.644444    0.002183    4.743591 4.910083 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001485       3.659378              True    True
        526         375    0.712928       375  0.640000    0.001858    4.617275 5.632499 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001160       3.533063              True    True
        526         375    0.712928       375  0.640000    0.001858    4.617275 5.632499 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001160       3.533063              True    True
        526         105    0.199620       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000325       4.750933              True    True
        526         526    1.000000       526  0.574144    0.000698    1.084212 1.566415 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        526         526    1.000000       526  0.574144    0.000698    1.084212 1.566415 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        526           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        526           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        526           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False