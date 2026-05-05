# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 565
- Baseline expectancy (all days): 0.000754
- Baseline sharpe_ann (all days): 1.172
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 160
- COILED: 115

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        565         290    0.513274       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001221       3.016832              True    True
        565         405    0.716814       405  0.632099    0.001728    4.201707 5.326635 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000974       3.029610              True    True
        565         405    0.716814       405  0.632099    0.001728    4.201707 5.326635 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000974       3.029610              True    True
        565         115    0.203540       115  0.626087    0.001104    6.029295 4.073007 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000350       4.857198              True    True
        565         565    1.000000       565  0.571681    0.000754    1.172097 1.755041 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        565         565    1.000000       565  0.571681    0.000754    1.172097 1.755041 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        565           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        565           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        565           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False