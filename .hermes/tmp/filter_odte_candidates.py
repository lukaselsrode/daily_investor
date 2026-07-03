#!/usr/bin/env python3
import contextlib
import json
import os
from datetime import datetime, timezone

import robin_stocks.robinhood as rb
TARGET='435050133'
EXP='2026-06-26'
BP=float(os.environ.get('BP','42.14'))
SPOTS={'SPY':733.9,'QQQ':711.4,'IWM':298.5,'MSFT':368.4,'MSTR':85.25,'HOOD':97.2,'COIN':148.0,'TSLA':383.2,'META':555.9}
SYMS=['SPY','QQQ','IWM','MSFT','MSTR','HOOD','COIN','TSLA','META']

def fl(x):
    try: return float(x)
    except Exception: return None

def compact(o):
    bid=fl(o.get('bid_price')); ask=fl(o.get('ask_price')); mark=fl(o.get('mark_price') or o.get('adjusted_mark_price'))
    strike=fl(o.get('strike_price')); vol=fl(o.get('volume')); oi=fl(o.get('open_interest'))
    spread=None
    if bid is not None and ask is not None: spread=ask-bid
    mid=None
    if bid is not None and ask is not None: mid=(bid+ask)/2
    return {
        'chain_symbol':o.get('chain_symbol'), 'id':o.get('id'), 'expiration_date':o.get('expiration_date'), 'type':o.get('type'), 'strike_price':strike,
        'bid_price':bid,'ask_price':ask,'mark_price':mark,'debit_ask_dollars':None if ask is None else round(ask*100,2),
        'spread':spread,'spread_pct_mid':None if not mid or spread is None else round(spread/mid,3),
        'delta':fl(o.get('delta')),'gamma':fl(o.get('gamma')),'implied_volatility':fl(o.get('implied_volatility')),
        'open_interest':oi,'volume':vol,'updated_at':o.get('updated_at'),'rhs_tradability':o.get('rhs_tradability') or o.get('tradability'),
        'sellout_datetime':o.get('sellout_datetime')
    }

with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
    rb.login(store_session=True)
rows=[]
errors=[]
for sym in SYMS:
    spot=SPOTS[sym]
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            opts=rb.options.find_options_by_expiration(sym, EXP, 'call')
    except Exception as e:
        errors.append(f'{sym}:{type(e).__name__}:{str(e)[:80]}'); continue
    for o in opts or []:
        c=compact(o)
        st=c['strike_price']; ask=c['ask_price']; bid=c['bid_price']
        if st is None or ask is None or bid is None: continue
        # near/OTM continuation calls, account-sized; include slightly ITM for IWM/MSFT if cheap enough
        if st < spot-1.0: continue
        if ask*100 > BP: continue
        if c['spread_pct_mid'] is not None and c['spread_pct_mid'] > 0.35: continue
        if (c['open_interest'] or 0) < 50 and (c['volume'] or 0) < 10: continue
        # avoid absurdly far OTM lottos unless no spread; cap distance: indexes 1%, singles 3%
        maxdist=0.012 if sym in ['SPY','QQQ','IWM'] else 0.035
        if (st-spot)/spot > maxdist: continue
        c['spot']=spot; c['distance_pct']=round((st-spot)/spot*100,3)
        rows.append(c)
rows.sort(key=lambda r:(r['chain_symbol'], abs(r['distance_pct']), r['debit_ask_dollars'] or 999))
print(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'expiration':EXP,'buying_power':BP,'candidate_count':len(rows),'candidates':rows[:40],'errors':errors},indent=2,sort_keys=True))
