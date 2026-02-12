# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 510
- Baseline expectancy (all days): 0.000733
- Baseline sharpe_ann (all days): 1.130
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 258
- UNRESOLVED_PRESSURE: 146
- COILED: 106

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        510         258    0.505882       258  0.647287    0.002262    4.900750 4.958749 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001529       3.770549              True    True
        510         364    0.713725       364  0.640110    0.001863    4.611711 5.542587 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001130       3.481510              True    True
        510         364    0.713725       364  0.640110    0.001863    4.611711 5.542587 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001130       3.481510              True    True
        510         106    0.207843       106  0.622642    0.000893    4.607247 2.988093 -0.014334                          PRESSURE_is_COILED          Only COILED pressure_state           0.000160       3.477047              True    True
        510         510    1.000000       510  0.574510    0.000733    1.130200 1.607831 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        510         510    1.000000       510  0.574510    0.000733    1.130200 1.607831 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        510           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        510           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        510           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False