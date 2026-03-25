# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 537
- Baseline expectancy (all days): 0.000614
- Baseline sharpe_ann (all days): 0.955
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 279
- UNRESOLVED_PRESSURE: 153
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        537         279    0.519553       279  0.630824    0.002002    4.311598 4.536700 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001388       3.356292              True    True
        537         384    0.715084       384  0.630208    0.001735    4.261557 5.260580 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001120       3.306251              True    True
        537         384    0.715084       384  0.630208    0.001735    4.261557 5.260580 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001120       3.306251              True    True
        537         105    0.195531       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000408       4.879840              True    True
        537         537    1.000000       537  0.567970    0.000614    0.955305 1.394534 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        537         537    1.000000       537  0.567970    0.000614    0.955305 1.394534 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        537           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        537           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        537           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False