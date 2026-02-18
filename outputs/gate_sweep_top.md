# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 513
- Baseline expectancy (all days): 0.000738
- Baseline sharpe_ann (all days): 1.140
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 261
- UNRESOLVED_PRESSURE: 147
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        513         261    0.508772       261  0.651341    0.002264    4.932134 5.019435 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001527       3.792212              True    True
        513         366    0.713450       366  0.644809    0.001908    4.768063 5.746220 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001170       3.628140              True    True
        513         366    0.713450       366  0.644809    0.001908    4.768063 5.746220 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001170       3.628140              True    True
        513         105    0.204678       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000285       4.695223              True    True
        513         513    1.000000       513  0.576998    0.000738    1.139922 1.626424 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        513         513    1.000000       513  0.576998    0.000738    1.139922 1.626424 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        513           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        513           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        513           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False