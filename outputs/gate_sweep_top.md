# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 508
- Baseline expectancy (all days): 0.000761
- Baseline sharpe_ann (all days): 1.174
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 257
- UNRESOLVED_PRESSURE: 146
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        508         257    0.505906       257  0.649805    0.002272    4.913294 4.961797 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001511       3.739521              True    True
        508         362    0.712598       362  0.643646    0.001910    4.747917 5.690588 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001148       3.574144              True    True
        508         362    0.712598       362  0.643646    0.001910    4.747917 5.690588 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001148       3.574144              True    True
        508         105    0.206693       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000261       4.661372              True    True
        508         508    1.000000       508  0.576772    0.000761    1.173773 1.666540 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        508         508    1.000000       508  0.576772    0.000761    1.173773 1.666540 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        508           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        508           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        508           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False