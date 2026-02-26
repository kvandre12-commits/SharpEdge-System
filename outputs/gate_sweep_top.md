# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 518
- Baseline expectancy (all days): 0.000750
- Baseline sharpe_ann (all days): 1.162
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 265
- UNRESOLVED_PRESSURE: 148
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        518         265    0.511583       265  0.652830    0.002307    5.042753 5.171188 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001557       3.881047              True    True
        518         370    0.714286       370  0.645946    0.001942    4.860392 5.889412 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001192       3.698687              True    True
        518         370    0.714286       370  0.645946    0.001942    4.860392 5.889412 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001192       3.698687              True    True
        518         105    0.202703       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000272       4.673440              True    True
        518         518    1.000000       518  0.577220    0.000750    1.161705 1.665561 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        518         518    1.000000       518  0.577220    0.000750    1.161705 1.665561 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        518           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        518           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        518           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False