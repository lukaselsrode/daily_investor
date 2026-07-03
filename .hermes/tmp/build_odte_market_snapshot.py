#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import yfinance as yf

TICKERS=['SPY','QQQ','IWM','VIXY','^VIX']
out={'generated_at':datetime.now(timezone.utc).isoformat(),'source':'yfinance_1m'}
now_et=datetime.now(ZoneInfo('America/New_York'))
close_et=now_et.replace(hour=16,minute=0,second=0,microsecond=0)
out['minutes_to_close']=max(0, int((close_et-now_et).total_seconds()//60))
for sym in TICKERS:
    try:
        df=yf.download(sym,period='1d',interval='1m',progress=False,auto_adjust=False,threads=False)
        if df is None or df.empty:
            continue
        if hasattr(df.columns, 'levels') and len(df.columns.levels)>1:
            # yfinance may return a multiindex even for one ticker
            df=df.xs(sym, axis=1, level=1, drop_level=True)
        df=df.dropna(subset=['Close'])
        close=df['Close']
        high=df['High']; low=df['Low']; vol=df['Volume'] if 'Volume' in df else None
        last=float(close.iloc[-1])
        prev=None
        try:
            prev=float(yf.Ticker(sym).fast_info.get('previous_close'))
        except Exception:
            pass
        if not prev:
            prev=float(close.iloc[0])
        vwap=None
        if vol is not None and float(vol.sum())>0:
            typical=(high+low+close)/3.0
            vwap=float((typical*vol).sum()/vol.sum())
        above_vwap = None if vwap is None else last >= vwap
        first=df.iloc[:30]
        orb_state='unknown'
        if len(first)>=5:
            orb_high=float(first['High'].max()); orb_low=float(first['Low'].min())
            if last>orb_high: orb_state='above'
            elif last<orb_low: orb_state='below'
            else: orb_state='inside'
        key=sym.lower().replace('^','')
        if sym in ['SPY','QQQ','IWM']:
            out[f'{key}_price']=last
            out[f'{key}_prev_close']=prev
            out[f'{key}_change_pct']=(last-prev)/prev*100 if prev else None
            out[f'{key}_vwap']=vwap
            out[f'{key}_above_vwap']=above_vwap
            out[f'{key}_orb_state']=orb_state
        elif sym=='VIXY':
            out['vixy']={'price':last,'prev_close':prev,'day_change_pct':(last-prev)/prev*100 if prev else None,'above_vwap':above_vwap,'vwap':vwap}
            out['vixy_change_pct']=out['vixy']['day_change_pct']
        elif sym=='^VIX':
            out['vix']=last; out['vix_prev_close']=prev; out['vix_change_pct']=(last-prev)/prev*100 if prev else None
    except Exception as e:
        out.setdefault('errors',[]).append(f'{sym}:{type(e).__name__}:{str(e)[:120]}')
# use SPY as broad gap/spot proxy
if out.get('spy_prev_close'):
    out['gap_pct']=out.get('spy_change_pct')
    out['spot']=out.get('spy_price')
# no gamma band yet; conservative expected move placeholder from observed day range if present
try:
    # do not score expected move from elapsed range; leave absent rather than fabricate
    pass
except Exception:
    pass
print(json.dumps(out,indent=2,sort_keys=True))
