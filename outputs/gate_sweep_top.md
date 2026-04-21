# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 555
- Baseline expectancy (all days): 0.000744
- Baseline sharpe_ann (all days): 1.149
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 289
- UNRESOLVED_PRESSURE: 156
- COILED: 110

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        555         289    0.520721       289  0.633218    0.001947    4.130466 4.423313 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001203       2.981076              True    True
        555         399    0.718919       399  0.631579    0.001710    4.146872 5.218033 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000966       2.997482              True    True
        555         399    0.718919       399  0.631579    0.001710    4.146872 5.218033 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000966       2.997482              True    True
        555         110    0.198198       110  0.627273    0.001089    6.086260 4.021116 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000345       4.936869              True    True
        555         555    1.000000       555  0.572973    0.000744    1.149390 1.705744 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        555         555    1.000000       555  0.572973    0.000744    1.149390 1.705744 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        555           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        555           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        555           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False