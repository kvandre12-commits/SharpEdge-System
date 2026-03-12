# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 528
- Baseline expectancy (all days): 0.000690
- Baseline sharpe_ann (all days): 1.074
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 272
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        528         272    0.515152       272  0.639706    0.002156    4.698978 4.881885 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001466       3.625343              True    True
        528         377    0.714015       377  0.636605    0.001840    4.582886 5.605437 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001151       3.509252              True    True
        528         377    0.714015       377  0.636605    0.001840    4.582886 5.605437 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001151       3.509252              True    True
        528         105    0.198864       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000333       4.761511              True    True
        528         528    1.000000       528  0.571970    0.000690    1.073634 1.554079 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        528         528    1.000000       528  0.571970    0.000690    1.073634 1.554079 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        528           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        528           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        528           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False