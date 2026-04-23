# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 558
- Baseline expectancy (all days): 0.000739
- Baseline sharpe_ann (all days): 1.144
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 158
- COILED: 110

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        558         290    0.519713       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001236       3.045051              True    True
        558         400    0.716846       400  0.632500    0.001731    4.194484 5.284553 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000992       3.050606              True    True
        558         400    0.716846       400  0.632500    0.001731    4.194484 5.284553 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000992       3.050606              True    True
        558         110    0.197133       110  0.627273    0.001089    6.086260 4.021116 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000350       4.942382              True    True
        558         558    1.000000       558  0.571685    0.000739    1.143878 1.702144 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        558         558    1.000000       558  0.571685    0.000739    1.143878 1.702144 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False