# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 551
- Baseline expectancy (all days): 0.000712
- Baseline sharpe_ann (all days): 1.098
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 288
- UNRESOLVED_PRESSURE: 155
- COILED: 108

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        551         288    0.522686       288  0.631944    0.001911    4.061263 4.341673 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001199       2.963017              True    True
        551         396    0.718693       396  0.631313    0.001702    4.117932 5.162100 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000990       3.019685              True    True
        551         396    0.718693       396  0.631313    0.001702    4.117932 5.162100 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000990       3.019685              True    True
        551         108    0.196007       108  0.629630    0.001145    6.116159 4.003966 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000432       5.017912              True    True
        551         551    1.000000       551  0.571688    0.000712    1.098246 1.623960 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        551         551    1.000000       551  0.571688    0.000712    1.098246 1.623960 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        551           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        551           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        551           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False