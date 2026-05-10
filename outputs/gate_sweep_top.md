# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 569
- Baseline expectancy (all days): 0.000796
- Baseline sharpe_ann (all days): 1.239
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 293
- UNRESOLVED_PRESSURE: 161
- COILED: 115

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        569         293    0.514938       293  0.634812    0.002000    4.250371 4.583111 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001203       3.011182              True    True
        569         408    0.717047       408  0.632353    0.001747    4.252356 5.410774 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000951       3.013167              True    True
        569         408    0.717047       408  0.632353    0.001747    4.252356 5.410774 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000951       3.013167              True    True
        569         115    0.202109       115  0.626087    0.001104    6.029295 4.073007 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000308       4.790106              True    True
        569         569    1.000000       569  0.572935    0.000796    1.239189 1.862058 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        569         569    1.000000       569  0.572935    0.000796    1.239189 1.862058 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        569           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        569           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        569           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False