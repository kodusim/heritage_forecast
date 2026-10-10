"""Matched daily LSTM/Transformer experiment; never changes application projects.

Uses normalized annual-sheet observations, excluding conflicting duplicate cells.
Daily means require 96/144 observations; right labels mean data is available
at the stated origin. No imputation. Forecasts are origin + predicted delta.
"""
from pathlib import Path
import sys, json, hashlib, copy, time, argparse
import numpy as np
import pandas as pd
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / 'artifacts/long_horizon'
CONFIG = dict(lookback_days=30, horizons_days=[7,30], seeds=[42,43,44],
              train_target_before='2023-01-01', validation_origin_from='2023-01-01',
              validation_target_before='2024-01-01', test_origin_from='2024-01-01',
              daily_minimum_observations=96, epochs=40, patience=7,
              batch_size=256, learning_rate=.001, width=64, layers=2, dropout=.2,
              environment=['internal_primary','internal_backup'],
              target='mean of day ending at origin+h minus mean of day ending at origin',
              selection='validation normalized delta MSE; no test selection',
              input_modes=['self','environment'])

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(tomb_filter=None):
    from .seed_data import SEED,manifest
    seed_paths={r['id']:SEED/r['file'] for r in manifest()['files']}
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=[]; datasets=[]; counts=[]
    for tomb,fid,eid in [(1,'F02','F03'),(2,'F04','F05'),(3,'F07','F08')]:
        if tomb_filter and tomb!=tomb_filter: continue
        paths=[seed_paths[i] for i in (fid,eid)]
        a,e=[pd.read_csv(p,index_col=0,parse_dates=True,float_precision='round_trip') for p in paths]
        for p in paths: provenance.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p)))
        raw=a.join(e)
        group=raw.resample('D',closed='left',label='right')
        daily=group.mean().where(group.count()>=96)
        daily.to_csv(OUT/f'tomb{tomb}_daily.csv',encoding='utf-8-sig')
        idx=daily.index; n=len(daily)
        for sensor in a:
            # Same origins for both architectures AND both input modes.
            ok=daily[[sensor]+list(e)].notna().all(axis=1).rolling(30).sum().eq(30)
            for h in [7,30]: ok &= daily[sensor].shift(-h).notna()
            t=np.flatnonzero(ok.to_numpy())
            splits={
                'train':t[idx[t+30]<pd.Timestamp('2023-01-01')],
                'validation':t[(idx[t]>=pd.Timestamp('2023-01-01')) & (idx[t+30]<pd.Timestamp('2024-01-01'))],
                'test':t[idx[t]>=pd.Timestamp('2024-01-01')],
            }
            row=dict(tomb=tomb,sensor=sensor,**{k:len(v) for k,v in splits.items()})
            row['included']=all(len(splits[k])>=v for k,v in [('train',100),('validation',30),('test',30)])
            counts.append(row)
            if row['included']:
                assert idx[splits['train'][-1]+30]<idx[splits['validation'][0]]
                assert idx[splits['validation'][-1]+30]<idx[splits['test'][0]]
                datasets.append((tomb,sensor,daily,splits))
    suffix=f'_tomb{tomb_filter}' if tomb_filter else ''
    pd.DataFrame(counts).to_csv(OUT/f'sample_counts{suffix}.csv',index=False)
    (OUT/f'config{suffix}.json').write_text(json.dumps(dict(CONFIG,provenance=provenance,
        torch=importlib.metadata.version('torch'),source_sha256=digest(Path(__file__)),
        architecture_source_sha256=digest(ROOT/'research/models.py')),indent=2),encoding='utf-8')
    print('COUNTS',counts,flush=True)
    return datasets

def train_one(kind,seed,x,y):
    import torch
    from .models import LSTMModel,TransformerModel
    torch.manual_seed(seed)
    net=(LSTMModel(x['train'].shape[-1],2) if kind=='lstm' else TransformerModel(x['train'].shape[-1],2))
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    loss_fn=torch.nn.MSELoss()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x['train'],y['train']),batch_size=256,shuffle=True)
    best=float('inf'); state=None; stale=0; logs=[]; start=time.time()
    for epoch in range(40):
        net.train(); total=0
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(net(xb),yb)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            total+=loss.item()*len(xb)
        net.eval()
        with torch.no_grad(): vl=loss_fn(net(x['validation']),y['validation']).item()
        logs.append(dict(epoch=epoch+1,train=total/len(x['train']),validation=vl))
        if vl<best:
            best=vl;state=copy.deepcopy(net.state_dict());stale=0;best_epoch=epoch+1
        else: stale+=1
        if stale>=7: break
    net.load_state_dict(state);net.eval()
    with torch.no_grad(): pred=net(x['test']).numpy()
    return pred,state,dict(best_epoch=best_epoch,epochs=len(logs),seconds=time.time()-start,
        parameters=sum(p.numel() for p in net.parameters()),validation_loss=best),logs

def run(datasets):
    import torch
    for tomb,sensor,daily,splits in datasets:
        values=daily[sensor].to_numpy();test=splits['test'];orig=values[test]
        actual=np.stack([values[test+h] for h in [7,30]],axis=1)
        for mode in CONFIG['input_modes']:
            inputs=[sensor]+(CONFIG['environment'] if mode=='environment' else [])
            arr=daily[inputs].to_numpy()
            fit=arr[daily.index<pd.Timestamp('2023-01-01')]
            mean=np.nanmean(fit,axis=0);std=np.nanstd(fit,axis=0);std[std<1e-10]=1.
            z=(arr-mean)/std
            x={k:torch.tensor(np.stack([z[t-29:t+1] for t in ix]),dtype=torch.float32) for k,ix in splits.items()}
            # Normalize each horizon with TRAIN delta std, preventing large-horizon loss dominance.
            delta={k:np.stack([values[ix+h]-values[ix] for h in [7,30]],axis=1) for k,ix in splits.items()}
            scale=np.std(delta['train'],axis=0);scale[scale<1e-10]=1.
            y={k:torch.tensor(v/scale,dtype=torch.float32) for k,v in delta.items()}
            for seed in CONFIG['seeds']:
                for kind in ['lstm','transformer']:
                    name=f't{tomb}_{sensor}_{mode}_{kind}_{seed}';folder=OUT/'runs'/name
                    if (folder/'predictions.csv').exists(): continue
                    folder.mkdir(parents=True,exist_ok=True)
                    print('START',name,flush=True)
                    pred,state,info,logs=train_one(kind,seed,x,y)
                    prediction=orig[:,None]+pred*scale
                    rows=[]
                    for j,h in enumerate([7,30]):
                        for i,t in enumerate(test):
                            rows.append(dict(tomb=tomb,sensor=sensor,mode=mode,model=kind,seed=seed,
                                horizon_days=h,origin_time=str(daily.index[t]),target_time=str(daily.index[t+h]),
                                origin=orig[i],actual=actual[i,j],prediction=prediction[i,j],
                                persistence=orig[i],drift=orig[i]+(values[t]-values[t-29])/29*h))
                    torch.save(state,folder/'weights.pt')
                    pd.DataFrame(logs).to_csv(folder/'loss.csv',index=False)
                    (folder/'metadata.json').write_text(json.dumps(dict(info,inputs=inputs,mean=mean.tolist(),std=std.tolist(),
                        delta_scale=scale.tolist(),train_indices=splits['train'].tolist(),validation_indices=splits['validation'].tolist(),
                        test_indices=test.tolist())),encoding='utf-8')
                    pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
                    print('DONE',name,info,flush=True)

def summarize():
    frames=[pd.read_csv(p) for p in (OUT/'runs').glob('*/predictions.csv')]
    df=pd.concat(frames,ignore_index=True);df.to_csv(OUT/'predictions_all.csv.gz',index=False)
    metrics=[]
    for keys,g in df.groupby(['tomb','sensor','mode','model','seed','horizon_days']):
        row=dict(zip(['tomb','sensor','mode','model','seed','horizon_days'],keys))
        actual=g.actual.to_numpy();pred=g.prediction.to_numpy();err=pred-actual
        base=g.persistence.to_numpy()-actual
        row.update(n=len(g),MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),
            baseline_MAE=float(np.abs(base).mean()),baseline_RMSE=float(np.sqrt(np.mean(base**2))),
            drift_MAE=float(np.abs(g.drift-actual).mean()),
            delta_direction_accuracy=float(np.mean(np.sign(pred-g.origin)==np.sign(actual-g.origin))),
            R2=float(1-np.sum(err**2)/np.sum((actual-actual.mean())**2)) if actual.std()>0 else np.nan)
        metrics.append(row)
    m=pd.DataFrame(metrics);m.to_csv(OUT/'metrics_by_seed.csv',index=False)
    s=m.groupby(['tomb','sensor','mode','model','horizon_days']).agg(
        MAE=('MAE','mean'),MAE_seed_std=('MAE','std'),RMSE=('RMSE','mean'),
        baseline_MAE=('baseline_MAE','first'),baseline_RMSE=('baseline_RMSE','first'),
        n=('n','first'),seeds=('seed','nunique')).reset_index()
    s['MAE_improvement_vs_persistence_pct']=100*(1-s.MAE/s.baseline_MAE)
    s.to_csv(OUT/'comparison.csv',index=False)
    pivot=s.pivot(index=['tomb','sensor','mode','horizon_days'],columns='model',values='MAE').reset_index()
    pivot['winner']=np.where(pivot.lstm<pivot.transformer,'lstm','transformer')
    pivot.to_csv(OUT/'winners.csv',index=False)
    print(pivot.groupby(['mode','horizon_days','winner']).size().to_string(),flush=True)
    print('BASELINE WINS',s.assign(better=s.MAE<s.baseline_MAE).groupby(['mode','horizon_days','model']).better.sum().to_string(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize',action='store_true');parser.add_argument('--tomb',type=int)
    parser.add_argument('--sensors',nargs='+');args=parser.parse_args()
    if not args.summarize:
        import torch
        torch.set_num_threads(1)
        datasets=prepare(args.tomb)
        if args.sensors: datasets=[d for d in datasets if d[1] in args.sensors]
        run(datasets)
    if not args.tomb: summarize()
