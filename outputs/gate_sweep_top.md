# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 536
- Baseline expectancy (all days): 0.000622
- Baseline sharpe_ann (all days): 0.966
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 278
- UNRESOLVED_PRESSURE: 153
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        536         278    0.518657       278  0.633094    0.002022    4.349421 4.568289 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001400       3.383361              True    True
        536         383    0.714552       383  0.631854    0.001748    4.292098 5.291377 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001126       3.326037              True    True
        536         383    0.714552       383  0.631854    0.001748    4.292098 5.291377 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001126       3.326037              True    True
        536         105    0.195896       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000401       4.869085              True    True
        536         536    1.000000       536  0.569030    0.000622    0.966060 1.408920 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        536         536    1.000000       536  0.569030    0.000622    0.966060 1.408920 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        536           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        536           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        536           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False