# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 560
- Baseline expectancy (all days): 0.000754
- Baseline sharpe_ann (all days): 1.168
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 158
- COILED: 112

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        560         290    0.517857       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001221       3.021374              True    True
        560         402    0.717857       402  0.634328    0.001746    4.236827 5.351228 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000993       3.069272              True    True
        560         402    0.717857       402  0.634328    0.001746    4.236827 5.351228 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000993       3.069272              True    True
        560         112    0.200000       112  0.633929    0.001154    6.351190 4.234127 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000400       5.183636              True    True
        560         560    1.000000       560  0.573214    0.000754    1.167555 1.740488 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        560         560    1.000000       560  0.573214    0.000754    1.167555 1.740488 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        560           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        560           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        560           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False