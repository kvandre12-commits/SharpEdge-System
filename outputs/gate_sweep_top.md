# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 568
- Baseline expectancy (all days): 0.000783
- Baseline sharpe_ann (all days): 1.218
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 292
- UNRESOLVED_PRESSURE: 161
- COILED: 115

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        568         292    0.514085       292  0.633562    0.001978    4.202696 4.523963 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001195       2.984439              True    True
        568         407    0.716549       407  0.631450    0.001731    4.213429 5.354669 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000948       2.995173              True    True
        568         407    0.716549       407  0.631450    0.001731    4.213429 5.354669 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000948       2.995173              True    True
        568         115    0.202465       115  0.626087    0.001104    6.029295 4.073007 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000321       4.811039              True    True
        568         568    1.000000       568  0.572183    0.000783    1.218256 1.828995 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        568         568    1.000000       568  0.572183    0.000783    1.218256 1.828995 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        568           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        568           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        568           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False