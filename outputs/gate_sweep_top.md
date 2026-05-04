# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 564
- Baseline expectancy (all days): 0.000762
- Baseline sharpe_ann (all days): 1.183
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 159
- COILED: 115

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        564         290    0.514184       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001213       3.005512              True    True
        564         405    0.718085       405  0.632099    0.001728    4.201707 5.326635 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000966       3.018290              True    True
        564         405    0.718085       405  0.632099    0.001728    4.201707 5.326635 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000966       3.018290              True    True
        564         115    0.203901       115  0.626087    0.001104    6.029295 4.073007 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000342       4.845878              True    True
        564         564    1.000000       564  0.572695    0.000762    1.183417 1.770422 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        564         564    1.000000       564  0.572695    0.000762    1.183417 1.770422 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        564           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        564           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        564           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False