# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 507
- Baseline expectancy (all days): 0.000768
- Baseline sharpe_ann (all days): 1.183
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 256
- UNRESOLVED_PRESSURE: 146
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        507         256    0.504931       256  0.652344    0.002291    4.949418 4.988544 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001523       3.766353              True    True
        507         361    0.712032       361  0.645429    0.001922    4.775973 5.716302 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001154       3.592908              True    True
        507         361    0.712032       361  0.645429    0.001922    4.775973 5.716302 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001154       3.592908              True    True
        507         105    0.207101       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000254       4.652080              True    True
        507         507    1.000000       507  0.577909    0.000768    1.183065 1.678079 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        507         507    1.000000       507  0.577909    0.000768    1.183065 1.678079 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        507           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        507           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        507           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False