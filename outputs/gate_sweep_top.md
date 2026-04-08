# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 546
- Baseline expectancy (all days): 0.000623
- Baseline sharpe_ann (all days): 0.963
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 286
- UNRESOLVED_PRESSURE: 154
- COILED: 106

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        546         286    0.523810       286  0.629371    0.001870    3.969890 4.229230 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001248       3.007211              True    True
        546         392    0.717949       392  0.630102    0.001650    3.996000 4.983888 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001028       3.033321              True    True
        546         392    0.717949       392  0.630102    0.001650    3.996000 4.983888 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001028       3.033321              True    True
        546         106    0.194139       106  0.632075    0.001058    6.012936 3.899773 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000435       5.050257              True    True
        546         546    1.000000       546  0.569597    0.000623    0.962680 1.417026 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        546         546    1.000000       546  0.569597    0.000623    0.962680 1.417026 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        546           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        546           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        546           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False