# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 574
- Baseline expectancy (all days): 0.000793
- Baseline sharpe_ann (all days): 1.237
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 295
- UNRESOLVED_PRESSURE: 163
- COILED: 116

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        574         295    0.513937       295  0.633898    0.002008    4.275931 4.626382 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001214       3.038657              True    True
        574         411    0.716028       411  0.632603    0.001756    4.282278 5.468844 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000962       3.045004              True    True
        574         411    0.716028       411  0.632603    0.001756    4.282278 5.468844 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000962       3.045004              True    True
        574         116    0.202091       116  0.629310    0.001115    6.106815 4.143272 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000321       4.869540              True    True
        574         574    1.000000       574  0.573171    0.000793    1.237274 1.867333 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        574         574    1.000000       574  0.573171    0.000793    1.237274 1.867333 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        574           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        574           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        574           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False