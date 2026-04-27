# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 559
- Baseline expectancy (all days): 0.000752
- Baseline sharpe_ann (all days): 1.164
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 158
- COILED: 111

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        559         290    0.518784       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001223       3.025093              True    True
        559         401    0.717352       401  0.633416    0.001746    4.231682 5.338078 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000994       3.067847              True    True
        559         401    0.717352       401  0.633416    0.001746    4.231682 5.338078 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000994       3.067847              True    True
        559         111    0.198569       111  0.630631    0.001149    6.295561 4.178262 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000397       5.131725              True    True
        559         559    1.000000       559  0.572451    0.000752    1.163836 1.733394 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        559         559    1.000000       559  0.572451    0.000752    1.163836 1.733394 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        559           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        559           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        559           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False