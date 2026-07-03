#!/usr/bin/env python3
import contextlib, json, os
from datetime import datetime, timezone
import robin_stocks.robinhood as rb
SYM=os.environ.get('SYM','SPY')
EXP=os.environ.get('EXP','2026-06-26')
SPOT=float(os.environ.get('SPOT','733.9'))
BAND=float(os.environ.get('BAND','8'))

def fl(x):
    try: return float(x)
    except Exception: return None

def row(o):
    return {
        'chain_symbol': o.get('chain_symbol') or SYM,
        'expiration_date': o.get('expiration_date') or EXP,
        'strike_price': fl(o.get('strike_price')),
        'type': o.get('type'),
        'bid_price': fl(o.get('bid_price')),
        'ask_price': fl(o.get('ask_price')),
        'mark_price': fl(o.get('mark_price') or o.get('adjusted_mark_price')),
        'implied_volatility': fl(o.get('implied_volatility')),
        'delta': fl(o.get('delta')),
        'gamma': fl(o.get('gamma')),
        'open_interest': fl(o.get('open_interest')),
        'volume': fl(o.get('volume')),
        'updated_at': o.get('updated_at'),
    }
with open(os.devnull,'w') as devnull, contextlib.redirect_stdout(devnull):
    rb.login(store_session=True)
rows=[]
for typ in ['call','put']:
    with open(os.devnull,'w') as devnull, contextlib.redirect_stdout(devnull):
        opts=rb.options.find_options_by_expiration(SYM, EXP, typ)
    for o in opts or []:
        st=fl(o.get('strike_price'))
        if st is None or abs(st-SPOT)>BAND: continue
        r=row(o)
        if r['gamma'] is None and r['open_interest'] is None: continue
        rows.append(r)
print(json.dumps({'underlying':SYM,'expiration':EXP,'spot':SPOT,'rows':rows,'generated_at':datetime.now(timezone.utc).isoformat()},indent=2))
