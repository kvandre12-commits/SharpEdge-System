# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 541
- Baseline expectancy (all days): 0.000549
- Baseline sharpe_ann (all days): 0.852
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 283
- UNRESOLVED_PRESSURE: 153
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        541         283    0.523105       283  0.625442    0.001859    3.928913 4.163565 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001309       3.076680              True    True
        541         388    0.717190       388  0.626289    0.001632    3.937470 4.885768 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001083       3.085237              True    True
        541         388    0.717190       388  0.626289    0.001632    3.937470 4.885768 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001083       3.085237              True    True
        541         105    0.194085       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000473       4.982912              True    True
        541         541    1.000000       541  0.565619    0.000549    0.852233 1.248696 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        541         541    1.000000       541  0.565619    0.000549    0.852233 1.248696 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        541           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        541           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        541           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False