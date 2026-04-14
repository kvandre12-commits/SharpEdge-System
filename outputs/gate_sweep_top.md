# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 550
- Baseline expectancy (all days): 0.000691
- Baseline sharpe_ann (all days): 1.066
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 288
- UNRESOLVED_PRESSURE: 155
- COILED: 107

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        550         288    0.523636       288  0.631944    0.001911    4.061263 4.341673 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001220       2.994945              True    True
        550         395    0.718182       395  0.630380    0.001676    4.061769 5.085264 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000984       2.995451              True    True
        550         395    0.718182       395  0.630380    0.001676    4.061769 5.085264 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000984       2.995451              True    True
        550         107    0.194545       107  0.626168    0.001041    5.939091 3.870006 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000350       4.872773              True    True
        550         550    1.000000       550  0.570909    0.000691    1.066318 1.575317 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        550         550    1.000000       550  0.570909    0.000691    1.066318 1.575317 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        550           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        550           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        550           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False