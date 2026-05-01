# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 563
- Baseline expectancy (all days): 0.000758
- Baseline sharpe_ann (all days): 1.177
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 159
- COILED: 114

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        563         290    0.515098       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001217       3.012056              True    True
        563         404    0.717584       404  0.631188    0.001725    4.190375 5.305707 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000967       3.013501              True    True
        563         404    0.717584       404  0.631188    0.001725    4.190375 5.305707 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000967       3.013501              True    True
        563         114    0.202487       114  0.622807    0.001090    5.932031 3.989841 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000331       4.755158              True    True
        563         563    1.000000       563  0.571936    0.000758    1.176873 1.759072 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        563         563    1.000000       563  0.571936    0.000758    1.176873 1.759072 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        563           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        563           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        563           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False