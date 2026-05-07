# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 567
- Baseline expectancy (all days): 0.000790
- Baseline sharpe_ann (all days): 1.228
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 291
- UNRESOLVED_PRESSURE: 161
- COILED: 115

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        567         291    0.513228       291  0.635739    0.001996    4.235559 4.551526 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001206       3.007674              True    True
        567         406    0.716049       406  0.633005    0.001743    4.239789 5.381546 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000953       3.011904              True    True
        567         406    0.716049       406  0.633005    0.001743    4.239789 5.381546 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000953       3.011904              True    True
        567         115    0.202822       115  0.626087    0.001104    6.029295 4.073007 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000314       4.801410              True    True
        567         567    1.000000       567  0.573192    0.000790    1.227885 1.841828 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        567         567    1.000000       567  0.573192    0.000790    1.227885 1.841828 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        567           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        567           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        567           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False