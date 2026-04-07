# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 545
- Baseline expectancy (all days): 0.000623
- Baseline sharpe_ann (all days): 0.962
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 285
- UNRESOLVED_PRESSURE: 154
- COILED: 106

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        545         285    0.522936       285  0.628070    0.001875    3.973808 4.225996 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001252       3.011495              True    True
        545         391    0.717431       391  0.629156    0.001654    3.998547 4.980699 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001031       3.036233              True    True
        545         391    0.717431       391  0.629156    0.001654    3.998547 4.980699 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001031       3.036233              True    True
        545         106    0.194495       106  0.632075    0.001058    6.012936 3.899773 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000435       5.050623              True    True
        545         545    1.000000       545  0.568807    0.000623    0.962314 1.415190 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        545         545    1.000000       545  0.568807    0.000623    0.962314 1.415190 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        545           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        545           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        545           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False