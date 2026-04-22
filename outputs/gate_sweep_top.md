# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 557
- Baseline expectancy (all days): 0.000748
- Baseline sharpe_ann (all days): 1.156
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 157
- COILED: 110

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        557         290    0.520646       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001227       3.033047              True    True
        557         400    0.718133       400  0.632500    0.001731    4.194484 5.284553 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000984       3.038602              True    True
        557         400    0.718133       400  0.632500    0.001731    4.194484 5.284553 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000984       3.038602              True    True
        557         110    0.197487       110  0.627273    0.001089    6.086260 4.021116 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000341       4.930378              True    True
        557         557    1.000000       557  0.572711    0.000748    1.155882 1.718466 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        557         557    1.000000       557  0.572711    0.000748    1.155882 1.718466 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        557           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        557           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        557           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False