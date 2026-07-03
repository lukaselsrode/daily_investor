import json, datetime
from zoneinfo import ZoneInfo
try:
    import yfinance as yf
except Exception as e:
    print(json.dumps({'ok': False, 'error': f'yfinance import failed: {e}'}))
    raise SystemExit(0)
syms = ['SPY', 'QQQ', 'IWM', 'MU', 'VIXY']
now_et = datetime.datetime.now(ZoneInfo('America/New_York'))
out = {'ok': True, 'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'now_et': now_et.isoformat(), 'symbols': {}}
for sym in syms:
    try:
        df = yf.download(sym, period='1d', interval='1m', progress=False, auto_adjust=False, prepost=False, threads=False)
        if df is None or df.empty:
            out['symbols'][sym] = {'ok': False, 'error': 'no bars'}
            continue
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        close = float(df['Close'].iloc[-1])
        volume_total = float(df['Volume'].sum())
        vwap = float((df['Close'] * df['Volume']).sum() / df['Volume'].sum()) if volume_total else None
        first = df.iloc[:30]
        orb_high = float(first['High'].max())
        orb_low = float(first['Low'].min())
        if close > orb_high:
            orb = 'above'
        elif close < orb_low:
            orb = 'below'
        else:
            orb = 'inside'
        out['symbols'][sym] = {
            'ok': True,
            'last': close,
            'vwap': vwap,
            'above_vwap': (close > vwap if vwap else None),
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_state': orb,
            'bars': int(len(df)),
        }
    except Exception as e:
        out['symbols'][sym] = {'ok': False, 'error': str(e)}
print(json.dumps(out, default=str))
