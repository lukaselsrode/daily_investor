#!/usr/bin/env python3
import json, os, sys, math
from datetime import datetime, timezone
TARGET=os.environ.get('RH_TARGET_ACCOUNT','435050133')
MASK='***'+TARGET[-4:]

def f(x):
    try:
        return float(x)
    except Exception:
        return None

def pick(d,*keys):
    for k in keys:
        if isinstance(d,dict) and d.get(k) not in (None,''):
            return d.get(k)
    return None

out={'ts':datetime.now(timezone.utc).isoformat(),'target_account_masked':MASK,'broker':{'ok':False,'account_verified':False,'errors':[]},'quotes':{},'option_positions':[],'open_option_orders':[]}
try:
    import robin_stocks.robinhood as rb
    try:
        # try token/cache based auth; do not print secrets
        login_res = rb.login(store_session=True)
        out['broker']['login_result_type']=type(login_res).__name__
    except Exception as e:
        out['broker']['errors'].append('login_failed:'+type(e).__name__+':'+str(e)[:120])
    # account profile for explicit target
    try:
        acct = rb.profiles.load_account_profile(account_number=TARGET)
        out['broker']['raw_account_type']=type(acct).__name__
        if isinstance(acct,dict):
            accnum = str(acct.get('account_number') or acct.get('account') or '')
            url = str(acct.get('url') or '')
            out['broker']['account_verified'] = (TARGET in accnum) or (TARGET in url)
            out['broker']['buying_power']=pick(acct,'buying_power','cash_available_for_withdrawal','portfolio_cash','uncleared_deposits')
            out['broker']['option_level']=pick(acct,'option_level','options_trading_level')
            out['broker']['agentic_allowed']=pick(acct,'agentic_allowed')
            out['broker']['account_masked']=MASK if out['broker']['account_verified'] else 'unverified'
            out['broker']['ok']=True
        else:
            out['broker']['errors'].append('account_profile_non_dict')
    except Exception as e:
        out['broker']['errors'].append('account_profile_failed:'+type(e).__name__+':'+str(e)[:120])
    # open option positions
    for method_name in ['get_open_option_positions','get_aggregate_open_positions']:
        try:
            method = getattr(rb.options, method_name)
            try:
                pos = method(account_number=TARGET)
            except TypeError:
                pos = method()
            if isinstance(pos,dict): pos=[pos]
            if pos:
                for p in pos:
                    if not isinstance(p,dict): continue
                    qty=f(pick(p,'quantity','chain_quantity','processed_quantity'))
                    if qty and abs(qty)>1e-9:
                        out['option_positions'].append({
                            'method':method_name,
                            'chain_symbol':pick(p,'chain_symbol','symbol'),
                            'option':pick(p,'option','option_id','id'),
                            'account_verified': TARGET in str(p.get('account','')) or TARGET in str(p.get('account_number','')) or TARGET in str(p.get('url','')),
                            'quantity':qty,
                            'average_price':pick(p,'average_price','average_price_paid'),
                            'type':pick(p,'type','option_type'),
                            'expiration_date':pick(p,'expiration_date'),
                            'strike_price':pick(p,'strike_price')})
            break
        except Exception as e:
            out['broker']['errors'].append(method_name+'_failed:'+type(e).__name__+':'+str(e)[:120])
    # open option orders
    try:
        try:
            orders = rb.orders.get_all_open_option_orders(account_number=TARGET)
        except TypeError:
            orders = rb.orders.get_all_open_option_orders()
        if isinstance(orders,dict): orders=[orders]
        for o in orders or []:
            if not isinstance(o,dict): continue
            state=str(o.get('state') or o.get('cancel_url') or '')
            out['open_option_orders'].append({
                'id':o.get('id'), 'state':o.get('state'), 'direction':o.get('direction'), 'opening_strategy':o.get('opening_strategy'), 'closing_strategy':o.get('closing_strategy'),
                'quantity':pick(o,'quantity','processed_quantity'), 'price':pick(o,'price','premium'), 'chain_symbol':pick(o,'chain_symbol'),
                'account_verified': TARGET in str(o.get('account','')) or TARGET in str(o.get('account_number','')) or TARGET in str(o.get('url',''))})
    except Exception as e:
        out['broker']['errors'].append('open_option_orders_failed:'+type(e).__name__+':'+str(e)[:120])
    # quotes from RH
    syms=['SPY','QQQ','IWM','VIXY','MSFT','MSTR','AMD','MU','TSLA','HOOD','COIN','AAPL','META','SMH']
    try:
        qs=rb.stocks.get_quotes(syms)
        for q in qs or []:
            if not isinstance(q,dict): continue
            sym=q.get('symbol')
            if sym:
                out['quotes'][sym]={'source':'robinhood','last_trade_price':q.get('last_trade_price'),'previous_close':q.get('previous_close'),'updated_at':q.get('updated_at'),'bid_price':q.get('bid_price'),'ask_price':q.get('ask_price')}
    except Exception as e:
        out['broker']['errors'].append('stock_quotes_failed:'+type(e).__name__+':'+str(e)[:120])
except Exception as e:
    out['broker']['errors'].append('robin_import_or_probe_failed:'+type(e).__name__+':'+str(e)[:160])
# fallback yfinance for missing broad quotes
try:
    import yfinance as yf
    need=[s for s in ['SPY','QQQ','IWM','VIXY','MSFT','MSTR','AMD','MU','TSLA','HOOD','COIN','AAPL','META','SMH'] if s not in out['quotes']]
    if need:
        data=yf.download(' '.join(need),period='1d',interval='1m',progress=False,auto_adjust=False,threads=True)
        import pandas as pd
        if not data.empty:
            for s in need:
                try:
                    close=data['Close'][s].dropna() if hasattr(data['Close'],'columns') else data['Close'].dropna()
                    if len(close): out['quotes'][s]={'source':'yfinance','last_trade_price':float(close.iloc[-1]),'updated_at':str(close.index[-1])}
                except Exception: pass
except Exception as e:
    out.setdefault('quote_errors',[]).append('yf_failed:'+type(e).__name__+':'+str(e)[:120])
print(json.dumps(out, indent=2, sort_keys=True))
