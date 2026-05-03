
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from portfolio_backtester import config_loader
from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.interfaces import create_data_source
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'dev/spy_seasonal_shared_exit_vectorized_ibkr_slip05_legacy'
ENTRY={3:16,4:11,5:14,6:20,7:7,10:19,11:15,12:14}
HOLD=10; START=pd.Timestamp('2005-01-01'); END=pd.Timestamp('2024-12-31')
PORT=100000.0; CPS=0.005; CMIN=1.0; CMAX=0.005; SLIP=0.5
MONTH={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

def nth_bday(y,m,n):
    b=pd.bdate_range(pd.Timestamp(y,m,1),(pd.Timestamp(y,m,1)+pd.offsets.MonthEnd(1)).normalize())
    i=n-1 if n>0 else n
    i=max(min(i,len(b)-1),-len(b))
    return pd.Timestamp(b[i])

def scenario():
    sp={'direction':'long','month_local_seasonal_windows':False,'hold_days':HOLD,'entry_day':1}
    for m in range(1,13): sp[f'trade_month_{m}']=m in ENTRY
    return {'name':'probe','strategy':'SeasonalSignalStrategy','start_date':str(START.date()),'end_date':str(END.date()),'benchmark_ticker':'SPY','rebalance_frequency':'ME','position_sizer':'equal_weight','timing_config':{'mode':'signal_based','scan_frequency':'D','min_holding_period':1,'trade_execution_timing':'bar_close'},'universe_config':{'type':'fixed','tickers':['SPY']},'train_window_months':36,'test_window_months':12,'extras':{'is_wfo':False,'risk_free_metrics_enabled':False},'strategy_params':sp}

def atr_series(h,l,c,lookback=21):
    prev=np.r_[c[0], c[:-1]]
    tr=np.maximum.reduce([h-l, np.abs(h-prev), np.abs(l-prev)])
    return pd.Series(tr).rolling(lookback, min_periods=1).mean().to_numpy()

def load_ohlc():
    config_loader.load_config(); gc=config_loader.GLOBAL_CONFIG
    gc.setdefault('data_source_config',{})['cache_only']=True
    gc.update({'portfolio_value':PORT,'commission_per_share':CPS,'commission_min_per_order':CMIN,'commission_max_percent_of_trade':CMAX,'slippage_bps':SLIP})
    canon=ScenarioNormalizer().normalize(scenario=scenario(), global_config=gc)
    ohlc,_,_=DataFetcher(gc, create_data_source(gc)).prepare_data_for_backtesting([canon], StrategyManager().get_strategy)
    close_s=ohlc.xs('Close', level='Field', axis=1)['SPY']; high_s=ohlc.xs('High', level='Field', axis=1)['SPY']; low_s=ohlc.xs('Low', level='Field', axis=1)['SPY']
    ix=pd.DatetimeIndex(close_s.index)
    loc_ix=pd.DatetimeIndex([pd.Timestamp(t).tz_convert(ix.tz).replace(tzinfo=None) for t in ix]) if ix.tz is not None else ix
    sel=(loc_ix>=START)&(loc_ix<=END)
    return ix[sel], loc_ix[sel], close_s.loc[sel].to_numpy(float), high_s.loc[sel].to_numpy(float), low_s.loc[sel].to_numpy(float)

def build_cycles(loc_ix, close, high, low, atr):
    cycles=[]
    for y in range(START.year-1, END.year+1):
        for m,n in ENTRY.items():
            ent=nth_bday(y,m,n); end=ent+BDay(HOLD-1)
            if end<START or ent>END: continue
            pos=np.where((loc_ix>=ent)&(loc_ix<=end))[0]
            if len(pos)==0: continue
            ei=np.where(loc_ix==ent)[0]
            ei=int(ei[0]) if len(ei) else None
            cycles.append({'entry':ent,'month':m,'pos':pos,'ei':ei,'entry_px':close[ei] if ei is not None else np.nan,'atr':atr[ei] if ei is not None else np.nan,'cl':close[pos],'ph':high[np.maximum(pos-1,0)],'pl':low[np.maximum(pos-1,0)]})
    cycles.sort(key=lambda x:x['entry'])
    resolved=np.full(len(close), -1, dtype=int)
    for ci,c in enumerate(cycles):
        for p in c['pos']:
            if resolved[p]<0 or c['entry']>cycles[resolved[p]]['entry']:
                resolved[p]=ci
    return cycles, resolved

def eval_params(cycles,resolved,close,rets,bench_rets,ix,ssl,stp,slm,tpm):
    target=np.zeros(len(close))
    for ci,c in enumerate(cycles):
        hit=np.zeros(len(c['pos']), dtype=bool)
        if ssl: hit |= c['cl'] < c['pl']
        if stp: hit |= c['cl'] > c['ph']
        if slm>0 and np.isfinite(c['entry_px']) and np.isfinite(c['atr']) and c['atr']>0: hit |= c['cl'] <= c['entry_px']-slm*c['atr']
        if tpm>0 and np.isfinite(c['entry_px']) and np.isfinite(c['atr']) and c['atr']>0: hit |= c['cl'] >= c['entry_px']+tpm*c['atr']
        active=c['pos'][:int(np.argmax(hit))] if hit.any() else c['pos']
        mask=resolved[active]==ci
        target[active[mask]]=1.0
    gross=target*rets
    delta=np.abs(target-np.r_[0.0,target[:-1]])
    tv=delta*PORT
    shares=np.where((tv>0)&np.isfinite(close)&(close>0), tv/close, 0.0)
    comm=np.zeros_like(tv); nz=shares>0
    comm[nz]=np.minimum(np.maximum(shares[nz]*CPS, CMIN), tv[nz]*CMAX)
    costs=(comm + tv*(SLIP/10000.0))/PORT
    pr=pd.Series(gross-costs,index=ix)
    f=calculate_optimizer_metrics_fast(pr, bench_rets, 'SPY', risk_free_rets=None, requested_metrics={'Sortino','Sharpe','Calmar','Total Return','Max Drawdown','Ann. Return'})
    return {'Sortino':float(f.get('Sortino',np.nan)),'Sharpe':float(f.get('Sharpe',np.nan)),'Calmar':float(f.get('Calmar',np.nan)),'Total Return':float(f.get('Total Return',np.nan)),'Max Drawdown':float(f.get('Max Drawdown',np.nan)),'Ann Return':float(f.get('Ann. Return',np.nan)),'Trades':int(np.count_nonzero(delta>1e-12)),'Total Cost':float(costs.sum())}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ix,loc_ix,close,high,low=load_ohlc()
    rets=np.r_[0.0, close[1:]/close[:-1]-1.0]
    bench=pd.Series(rets,index=ix)
    cycles,resolved=build_cycles(loc_ix, close, high, low, atr_series(high,low,close))
    rows=[]; best=None; t0=time.perf_counter()
    stops=[round(i*0.1,6) for i in range(51)]; tps=[round(i*0.25,6) for i in range(41)]
    for ssl in (False,True):
      for stp in (False,True):
       for slm in stops:
        for tpm in tps:
          r={'simple_high_low_stop_loss':ssl,'simple_high_low_take_profit':stp,'stop_loss_atr_multiple':slm,'take_profit_atr_multiple':tpm}
          r.update(eval_params(cycles,resolved,close,rets,bench,ix,ssl,stp,slm,tpm))
          rows.append(r)
          if not math.isnan(r['Sortino']) and (best is None or r['Sortino']>best['Sortino']): best=r
    df=pd.DataFrame(rows).sort_values('Sortino',ascending=False)
    df.to_csv(OUT/'shared_exit_grid_metrics.csv',index=False)
    summary={'best_row':best,'top_25':df.head(25).to_dict(orient='records'),'elapsed_seconds':time.perf_counter()-t0,'n_evaluations':len(rows),'period':{'start':str(START.date()),'end':str(END.date())},'hold_days':HOLD,'entry_by_month':{str(k):v for k,v in ENTRY.items()},'trade_months':[MONTH[m] for m in sorted(ENTRY)],'costs':{'portfolio_value':PORT,'commission_per_share':CPS,'commission_min_per_order':CMIN,'commission_max_percent_of_trade':CMAX,'slippage_bps':SLIP},'sortino_mode':'legacy_raw_returns_rf_off','method':'single-asset vectorized evaluator; bar_close target weights affect same close-to-close return period'}
    (OUT/'shared_exit_grid_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=False)); print('OUT',OUT)
if __name__=='__main__': main()
