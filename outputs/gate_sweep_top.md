# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 517
- Baseline expectancy (all days): 0.000735
- Baseline sharpe_ann (all days): 1.138
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 264
- UNRESOLVED_PRESSURE: 148
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        517         264    0.510638       264  0.651515    0.002284    4.989287 5.106698 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001548       3.851093              True    True
        517         369    0.713733       369  0.644986    0.001925    4.816679 5.828551 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001190       3.678484              True    True
        517         369    0.713733       369  0.644986    0.001925    4.816679 5.828551 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001190       3.678484              True    True
        517         105    0.203095       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000287       4.696950              True    True
        517         517    1.000000       517  0.576402    0.000735    1.138195 1.630278 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        517         517    1.000000       517  0.576402    0.000735    1.138195 1.630278 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        517           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        517           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        517           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False