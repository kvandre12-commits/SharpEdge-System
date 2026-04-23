# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 558
- Baseline expectancy (all days): 0.000739
- Baseline sharpe_ann (all days): 1.144
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 157
- COILED: 111

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        558         290    0.519713       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001236       3.045051              True    True
        558         401    0.718638       401  0.630923    0.001717    4.161973 5.250143 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000978       3.018095              True    True
        558         401    0.718638       401  0.630923    0.001717    4.161973 5.250143 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000978       3.018095              True    True
        558         111    0.198925       111  0.621622    0.001044    5.782813 3.837959 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000305       4.638936              True    True
        558         558    1.000000       558  0.571685    0.000739    1.143878 1.702144 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        558         558    1.000000       558  0.571685    0.000739    1.143878 1.702144 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        558           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False