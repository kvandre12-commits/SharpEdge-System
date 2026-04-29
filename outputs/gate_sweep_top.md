# Gate sweep (top 10)

- DB: `data/spy_truth.db`
- Symbol: `SPY`
- Days: 561
- Baseline expectancy (all days): 0.000744
- Baseline sharpe_ann (all days): 1.153
- MIN_TRADES: 30

## pressure_state counts (top 20)

- NORMAL: 290
- UNRESOLVED_PRESSURE: 158
- COILED: 113

## Top gates

 total_days  trade_days  trade_freq  n_trades  win_rate  expectancy  sharpe_ann   t_stat    max_dd                                   gate_name                               notes  uplift_expectancy  uplift_sharpe  meets_min_trades  usable
        561         290    0.516934       290  0.634483    0.001975    4.188929 4.493676 -0.082516                          PRESSURE_is_NORMAL          Only NORMAL pressure_state           0.001231       3.036173              True    True
        561         403    0.718360       403  0.632754    0.001730    4.196920 5.307413 -0.082516                     PRESSURE_not_UNRESOLVED    Exclude only unresolved pressure           0.000986       3.044163              True    True
        561         403    0.718360       403  0.632754    0.001730    4.196920 5.307413 -0.082516                   PRESSURE_NORMAL_or_COILED                     NORMAL + COILED           0.000986       3.044163              True    True
        561         113    0.201426       113  0.628319    0.001101    5.970253 3.997898 -0.008737                          PRESSURE_is_COILED          Only COILED pressure_state           0.000357       4.817497              True    True
        561         561    1.000000       561  0.572193    0.000744    1.152756 1.719961 -0.189989                           BASELINE_ALL_DAYS                No gating: every day           0.000000       0.000000              True    True
        561         561    1.000000       561  0.572193    0.000744    1.152756 1.719961 -0.189989                          REGIME_not_UNKNOWN         Any day with a regime_label           0.000000       0.000000              True    True
        561           0    0.000000         0       NaN         NaN         NaN      NaN       NaN                     CURRENT_trade_gate_eq_1 What your pipeline currently allows                NaN            NaN             False   False
        561           0    0.000000         0       NaN         NaN         NaN      NaN       NaN           REGIME_COILED_and_PRESSURE_COILED              High selectivity combo                NaN            NaN             False   False
        561           0    0.000000         0       NaN         NaN         NaN      NaN       NaN REGIME_COILED_and_PRESSURE_NORMAL_or_COILED  Coiled regime; pressure not broken                NaN            NaN             False   False