# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 516
- Baseline expectancy (all days): 0.000723
- Baseline sharpe_ann (all days): 1.118
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 263
- UNRESOLVED_PRESSURE: 148
- COILED: 105

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        516         263    0.509690       263  0.650190    0.002265    4.942904 5.049633 -0.060020                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001542       3.824953              True    True
        516         368    0.713178       368  0.644022    0.001910    4.778465 5.774468 -0.057760                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.001188       3.660513              True    True
        516         368    0.713178       368  0.644022    0.001910    4.778465 5.774468 -0.057760                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.001188       3.660513              True    True
        516         105    0.203488       105  0.628571    0.001023    5.835145 3.766570 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000300       4.717194              True    True
        516         516    1.000000       516  0.575581    0.000723    1.117951 1.599733 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        516         516    1.000000       516  0.575581    0.000723    1.117951 1.599733 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        516           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        516           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        516           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False