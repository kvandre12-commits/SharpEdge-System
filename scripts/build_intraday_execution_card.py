#!/usr/bin/env python3
import os,json,sqlite3
from datetime import datetime
DB_PATH=os.getenv('SPY_DB_PATH','data/spy_truth.db')
os.makedirs('outputs',exist_ok=True)
con=sqlite3.connect(DB_PATH)

def latest(table,col='session_date'):
    try:
        cur=con.execute(f'SELECT * FROM {table} ORDER BY {col} DESC LIMIT 1')
        cols=[x[0] for x in cur.description]
        row=cur.fetchone()
        return dict(zip(cols,row)) if row else {}
    except Exception:
        return {}

execution=latest('execution_state_daily')
options=latest('options_positioning_metrics')
con.close()

trend=float(execution.get('prob_trend_fused',0.5) or 0.5)
compression=int(execution.get('compression_flag',0) or 0)
exhaustion=float(execution.get('exhaustion_score',0.45) or 0.45)
wick=float(execution.get('upper_wick_ratio',0.3) or 0.3)
dealer=options.get('dealer_state_hint','UNKNOWN')

if compression and dealer in ('LONG_GAMMA','PINNED'):
    regime='PIN_COMPRESSION'
elif trend>=0.7 and exhaustion<0.4:
    regime='TREND_EXPANSION'
elif exhaustion>=0.7:
    regime='EXPANSION_EXHAUSTION'
elif dealer in ('DEFENSIVE','SHORT_GAMMA','UNWIND_RISK'):
    regime='VOLATILE_UNWIND'
else:
    regime='BALANCED_ROTATION'

if wick>0.6 and exhaustion>0.65:
    auction='FAILED_ACCEPTANCE'
elif trend>0.7 and exhaustion<0.3:
    auction='INITIATIVE_BUYING'
elif exhaustion>0.75:
    auction='RESPONSIVE_SELLING'
else:
    auction='BALANCE'

card=f'''\nSHARPEDGE EXECUTION CARD\n==============================\nTIME: {datetime.utcnow().isoformat()} UTC\nSPOT: {options.get('spot',0)}\nACTIVE WALL: {options.get('max_total_oi_strike','N/A')}\nWALL DRIFT: {options.get('wall_drift',0)}\n\nREGIME: {regime}\nDEALER: {dealer}\nAUCTION: {auction}\n\nTREND PROB: {trend:.2f}\nEXHAUSTION: {exhaustion:.2f}\nGAMMA CONCENTRATION: {float(options.get('gamma_concentration',0.4) or 0.4):.2f}\n==============================\n'''

payload={'regime':regime,'dealer':dealer,'auction':auction,'trend_prob':trend,'card':card}

with open('outputs/intraday_execution_card.txt','w') as f:
    f.write(card)

with open('outputs/intraday_execution_card.json','w') as f:
    json.dump(payload,f,indent=2)

print(card)
