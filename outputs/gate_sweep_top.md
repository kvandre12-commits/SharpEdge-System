# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 531
- Baseline expectancy (all days): 0.000666
- Baseline sharpe_ann (all days): 1.036
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 275
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        531         275    0.517891       275  0.636364    0.002094    4.522011 4.723867 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001428       3.486366              True    True
        531         380    0.715631       380  0.634211    0.001798    4.434819 5.445872 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001132       3.399175              True    True
        531         380    0.715631       380  0.634211    0.001798    4.434819 5.445872 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001132       3.399175              True    True
        531         105    0.197740       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000357       4.799500              True    True
        531         531    1.000000       531  0.570621    0.000666    1.035645 1.503342 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        531         531    1.000000       531  0.570621    0.000666    1.035645 1.503342 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        531           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        531           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        531           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False