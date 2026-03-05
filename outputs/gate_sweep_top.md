# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 523
- Baseline expectancy (all days): 0.000721
- Baseline sharpe_ann (all days): 1.120
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 268
- UNRESOLVED_PRESSURE: 150
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        523         268    0.512428       268  0.649254    0.002269    4.963562 5.118711 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001548       3.843842              True    True
        523         373    0.713193       373  0.643432    0.001918    4.798057 5.837402 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001197       3.678337              True    True
        523         373    0.713193       373  0.643432    0.001918    4.798057 5.837402 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001197       3.678337              True    True
        523         105    0.200765       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000302       4.715425              True    True
        523         523    1.000000       523  0.575526    0.000721    1.119720 1.613095 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        523         523    1.000000       523  0.575526    0.000721    1.119720 1.613095 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        523           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        523           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        523           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False