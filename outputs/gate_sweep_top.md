# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 515
- Baseline expectancy (all days): 0.000744
- Baseline sharpe_ann (all days): 1.151
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 263
- UNRESOLVED_PRESSURE: 147
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        515         263    0.510680       263  0.650190    0.002265    4.942809 5.049535 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001521       3.791899              True    True
        515         368    0.714563       368  0.644022    0.001910    4.778390 5.774379 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001166       3.627480              True    True
        515         368    0.714563       368  0.644022    0.001910    4.778390 5.774379 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001166       3.627480              True    True
        515         105    0.203883       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000279       4.684235              True    True
        515         515    1.000000       515  0.576699    0.000744    1.150910 1.645298 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        515         515    1.000000       515  0.576699    0.000744    1.150910 1.645298 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        515           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        515           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        515           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False