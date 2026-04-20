# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 554
- Baseline expectancy (all days): 0.000749
- Baseline sharpe_ann (all days): 1.156
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 289
- UNRESOLVED_PRESSURE: 156
- COILED: 109

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        554         289    0.521661       289  0.633218    0.001947    4.130466 4.423313 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001198       2.974394              True    True
        554         398    0.718412       398  0.633166    0.001720    4.165918 5.235425 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000971       3.009846              True    True
        554         398    0.718412       398  0.633166    0.001720    4.165918 5.235425 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000971       3.009846              True    True
        554         109    0.196751       109  0.633028    0.001117    6.250228 4.110634 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000368       5.094156              True    True
        554         554    1.000000       554  0.574007    0.000749    1.156072 1.714113 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        554         554    1.000000       554  0.574007    0.000749    1.156072 1.714113 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        554           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        554           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        554           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False