# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 534
- Baseline expectancy (all days): 0.000636
- Baseline sharpe_ann (all days): 0.990
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 278
- UNRESOLVED_PRESSURE: 151
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        534         278    0.520599       278  0.633094    0.002022    4.349421 4.568289 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001385       3.358988              True    True
        534         383    0.717228       383  0.631854    0.001748    4.292098 5.291377 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001111       3.301665              True    True
        534         383    0.717228       383  0.631854    0.001748    4.292098 5.291377 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001111       3.301665              True    True
        534         105    0.196629       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000386       4.844712              True    True
        534         534    1.000000       534  0.569288    0.000636    0.990433 1.441768 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        534         534    1.000000       534  0.569288    0.000636    0.990433 1.441768 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        534           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        534           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        534           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False