# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 570
- Baseline expectancy (all days): 0.000801
- Baseline sharpe_ann (all days): 1.248
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 293
- UNRESOLVED_PRESSURE: 161
- COILED: 116

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        570         293    0.514035       293  0.634812    0.002000    4.250371 4.583111 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001198       3.002644              True    True
        570         409    0.717544       409  0.633252    0.001752    4.267927 5.437238 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000951       3.020200              True    True
        570         409    0.717544       409  0.633252    0.001752    4.267927 5.437238 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000951       3.020200              True    True
        570         116    0.203509       116  0.629310    0.001126    6.152449 4.174234 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000324       4.904722              True    True
        570         570    1.000000       570  0.573684    0.000801    1.247727 1.876535 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        570         570    1.000000       570  0.573684    0.000801    1.247727 1.876535 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        570           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        570           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        570           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False