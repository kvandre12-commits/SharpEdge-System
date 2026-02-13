# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 511
- Baseline expectancy (all days): 0.000728
- Baseline sharpe_ann (all days): 1.122
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 259
- UNRESOLVED_PRESSURE: 147
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        511         259    0.506849       259  0.648649    0.002256    4.896792 4.964337 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001529       3.774516              True    True
        511         364    0.712329       364  0.642857    0.001900    4.737168 5.693367 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001173       3.614892              True    True
        511         364    0.712329       364  0.642857    0.001900    4.737168 5.693367 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001173       3.614892              True    True
        511         105    0.205479       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000295       4.712869              True    True
        511         511    1.000000       511  0.575342    0.000728    1.122276 1.598122 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        511         511    1.000000       511  0.575342    0.000728    1.122276 1.598122 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        511           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        511           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        511           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False