"""Small patch Transformer search using 2022/2023 only; 2024 retrospective audit.

All selection is saved before any 2024 prediction. This does not make 2024
an untouched test: its results were inspected in the preceding experiment.
"""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/long_horizon'
OUT=ROOT/'artifacts/optimized_transformer'
H=[7,30]
CANDIDATES=[dict(name=f'p{days}_{mode}_w{width}',days=days,mode=mode,width=width)
    for days,mode,width in [(30,'self',16),(90,'self',16),(180,'self',16),
        (30,'environment',16),(90,'environment',16),(180,'environment',16),
        (90,'self',32),(90,'environment',32)]]
POLICY=dict(candidates=CANDIDATES,search_seed=42,final_seeds=[42,43,44],
    selection_years=[2022,2023],max_epochs=50,patience=8,learning_rate=.0005,
    weight_decay=.01,batch_size=128,loss='SmoothL1',patch_days=5,
    layers=1,dropout=.15,center_inputs_at_origin=True,input_clip=10,
    objective='equal-weight mean of per-year/per-horizon validation MAE divided by persistence MAE',
    missing='zero placeholder accompanied by observed-value mask; no interpolation',
    shrinkage_grid=[0,.25,.5,.75,1],
    caveat='2024 was inspected previously; retrospective audit, not untouched external test')

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

class PatchTransformer(nn.Module):
    def __init__(self,features,days,width):
        super().__init__();self.features=features;self.days=days
        self.proj=nn.Linear(features*10,width)
        pos=torch.arange(days//5).float()[:,None]
        freq=torch.exp(torch.arange(0,width,2).float()*(-np.log(10000)/width))
        pe=torch.zeros(days//5,width);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe)
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=.15,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width*2,2)
        # Start with zero change, instead of arbitrary motion from random output weights.
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        b=x.shape[0]
        z=x.reshape(b,self.days//5,5*x.shape[-1])
        z=self.encoder(self.proj(z)+self.position)
        z=self.norm(z)
        return self.head(torch.cat([z[:,-1],z.mean(dim=1)],dim=1))

def origins(d,sensor,year):
    good=d[[sensor,'internal_primary','internal_backup']].notna().all(axis=1).rolling(30).sum().eq(30)
    for h in H:good &= d[sensor].shift(-h).notna()
    t=np.flatnonzero(good.to_numpy());idx=d.index
    train=t[idx[t+30]<pd.Timestamp(f'{year}-01-01')]
    validation=t[(idx[t]>=pd.Timestamp(f'{year}-01-01'))&(idx[t+30]<pd.Timestamp(f'{year+1}-01-01'))]
    return train,validation

def build_inputs(d,indices,inputs,days,feature_scale):
    """Only slice rows at or before each origin, including explicit missing masks."""
    arr=d[inputs].to_numpy();rows=[]
    for t in indices:
        start=max(0,t-days+1)
        w=np.full((days,len(inputs)),np.nan)
        segment=arr[start:t+1];w[-len(segment):]=segment
        observed=np.isfinite(w)
        normalized=np.clip((w-arr[t])/feature_scale,-10,10)
        rows.append(np.concatenate([np.nan_to_num(normalized,nan=0),observed.astype(float)],axis=1))
    return torch.tensor(np.asarray(rows),dtype=torch.float32)

def prepare_arrays(d,sensor,cfg,year):
    tr,va=origins(d,sensor,year)
    assert len(tr)>=100 and len(va)>=30
    inputs=[sensor]+(['internal_primary','internal_backup'] if cfg['mode']=='environment' else [])
    values=d[sensor].to_numpy()
    delta={k:np.stack([values[ix+h]-values[ix] for h in H],axis=1) for k,ix in [('train',tr),('validation',va)]}
    ys=np.quantile(np.abs(delta['train']),.75,axis=0);ys=np.maximum(ys,1e-8)
    fs=np.nanstd(d.loc[d.index<pd.Timestamp(f'{year}-01-01'),inputs].to_numpy(),axis=0)
    # Behavior features use typical 30-day change, rather than absolute long-term level variance.
    fs[0]=ys[1];fs=np.maximum(fs,1e-8)
    x={k:build_inputs(d,ix,inputs,cfg['days'],fs) for k,ix in [('train',tr),('validation',va)]}
    y={k:torch.tensor(v/ys,dtype=torch.float32) for k,v in delta.items()}
    meta=dict(sensor=sensor,config=cfg,year=year,inputs=inputs,feature_scale=fs.tolist(),target_scale=ys.tolist(),
        train_indices=tr.tolist(),validation_indices=va.tolist())
    return x,y,delta,meta

def fit(d,sensor,cfg,year,seed,folder):
    if (folder/'complete.json').exists(): return load_json(folder/'complete.json')
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    x,y,delta,meta=prepare_arrays(d,sensor,cfg,year)
    net=PatchTransformer(len(meta['inputs']),cfg['days'],cfg['width'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
    loss_fn=nn.SmoothL1Loss()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x['train'],y['train']),batch_size=128,shuffle=True)
    base=np.maximum(np.abs(delta['validation']).mean(axis=0),1e-10)
    ys=np.array(meta['target_scale']);best=1.;state=copy.deepcopy(net.state_dict());best_epoch=0;stale=0;logs=[]
    started=time.time()
    for epoch in range(1,51):
        net.train();total=0
        for xb,yb in loader:
            optimizer.zero_grad();loss=loss_fn(net(xb),yb);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();total+=loss.item()*len(xb)
        net.eval()
        with torch.no_grad():pred=net(x['validation']).numpy()*ys
        score=float((np.abs(pred-delta['validation']).mean(axis=0)/base).mean())
        logs.append(dict(epoch=epoch,train_loss=total/len(x['train']),validation_relative_MAE=score))
        if score<best-1e-5:best=score;best_epoch=epoch;state=copy.deepcopy(net.state_dict());stale=0
        else:stale+=1
        if stale>=8:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():pred=net(x['validation']).numpy()*ys
    torch.save(state,folder/'weights.pt');save_json(folder/'metadata.json',meta)
    pd.DataFrame(logs).to_csv(folder/'loss.csv',index=False)
    rows=[]
    for j,h in enumerate(H):
        for i,t in enumerate(meta['validation_indices']):rows.append(dict(origin=str(d.index[t]),horizon_days=h,actual_delta=delta['validation'][i,j],predicted_delta=pred[i,j]))
    pd.DataFrame(rows).to_csv(folder/'validation.csv',index=False)
    result=dict(seed=seed,year=year,best_epoch=best_epoch,epochs=len(logs),score=best,
        parameters=sum(p.numel() for p in net.parameters()),seconds=time.time()-started)
    save_json(folder/'complete.json',result);return result

def datasets(tomb):
    d=pd.read_csv(BASE/f'tomb{tomb}_daily.csv',index_col=0,parse_dates=True)
    counts=pd.read_csv(BASE/'sample_counts.csv')
    return d,counts[(counts.tomb==tomb)&counts.included].sensor.tolist()

def search(tomb,only_sensors=None):
    d,sensors=datasets(tomb)
    if only_sensors:sensors=[s for s in sensors if s in only_sensors]
    for sensor in sensors:
        folder=OUT/'sensors'/sensor
        if (folder/'selection.json').exists():continue
        trials=[]
        for cfg in CANDIDATES:
            scores=[]
            for year in [2022,2023]:
                dest=folder/'search'/cfg['name']/str(year)
                result=fit(d,sensor,cfg,year,42,dest);scores.append(result['score'])
                print('FIT',sensor,cfg['name'],year,result,flush=True)
            trials.append(dict(config=cfg,score=float(np.mean(scores)),year_scores=scores))
        selected=min(trials,key=lambda x:x['score']);cfg=selected['config']
        # Optional operational shrinkage is reported separately from pure network results.
        alpha={}
        for h in H:
            scores=[]
            for a in [0,.25,.5,.75,1]:
                per_year=[]
                for year in [2022,2023]:
                    v=pd.read_csv(folder/'search'/cfg['name']/str(year)/'validation.csv')
                    v=v[v.horizon_days==h];base=max(float(v.actual_delta.abs().mean()),1e-10)
                    per_year.append(float((v.actual_delta-a*v.predicted_delta).abs().mean()/base))
                scores.append((float(np.mean(per_year)),a))
            score,a=min(scores)
            alpha[str(h)]=a if score<.98 else 0.
        save_json(folder/'selection.json',dict(selected=selected,trials=trials,alpha=alpha,
            rule='selected using 2022 and 2023 validation only; no 2024 values or scores',
            policy_sha256=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()))
        print('SELECTED',sensor,cfg,alpha,flush=True)

def final(tomb):
    d,sensors=datasets(tomb)
    for sensor in sensors:
        folder=OUT/'sensors'/sensor;selection=load_json(folder/'selection.json');cfg=selection['selected']['config']
        # Same test origins as original experiment, including the same missingness restrictions.
        old=next((BASE/'runs').glob(f't{tomb}_{sensor}_self_lstm_42/metadata.json'))
        test=np.array(load_json(old)['test_indices']);values=d[sensor].to_numpy()
        for seed in [42,43,44]:
            dest=folder/'final'/str(seed)
            if (dest/'predictions.csv').exists():continue
            if seed==42:
                source=folder/'search'/cfg['name']/'2023'
                dest.mkdir(parents=True,exist_ok=True)
                import shutil
                for name in ['weights.pt','metadata.json','complete.json','loss.csv','validation.csv']:shutil.copy2(source/name,dest/name)
            else:fit(d,sensor,cfg,2023,seed,dest)
            meta=load_json(dest/'metadata.json')
            x=build_inputs(d,test,meta['inputs'],cfg['days'],np.array(meta['feature_scale']))
            net=PatchTransformer(len(meta['inputs']),cfg['days'],cfg['width'])
            net.load_state_dict(torch.load(dest/'weights.pt',weights_only=True));net.eval()
            with torch.no_grad():delta=net(x).numpy()*np.array(meta['target_scale'])
            rows=[]
            for j,h in enumerate(H):
                for i,t in enumerate(test):rows.append(dict(tomb=tomb,sensor=sensor,seed=seed,horizon_days=h,
                    origin_time=str(d.index[t]),target_time=str(d.index[t+h]),origin=values[t],actual=values[t+h],
                    optimized=values[t]+delta[i,j],guarded=values[t]+selection['alpha'][str(h)]*delta[i,j],
                    alpha=selection['alpha'][str(h)],config=cfg['name']))
            pd.DataFrame(rows).to_csv(dest/'predictions.csv',index=False)
            print('FINAL',sensor,seed,cfg['name'],flush=True)

def summarize():
    files=list((OUT/'sensors').glob('*/final/*/predictions.csv'));assert len(files)==66,len(files)
    pred=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    rows=[]
    for keys,g in pred.groupby(['tomb','sensor','seed','horizon_days']):
        common=dict(zip(['tomb','sensor','seed','horizon_days'],keys))
        for model in ['optimized','guarded']:
            err=g[model]-g.actual
            rows.append(dict(common,model=model,n=len(g),MAE=float(err.abs().mean()),RMSE=float(np.sqrt((err**2).mean())),
                baseline_MAE=float((g.origin-g.actual).abs().mean())))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics_by_seed.csv',index=False)
    avg=metrics.groupby(['tomb','sensor','horizon_days','model']).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),
        MAE_seed_std=('MAE','std'),baseline_MAE=('baseline_MAE','first'),n=('n','first')).reset_index()
    avg.to_csv(OUT/'metrics.csv',index=False)
    old=pd.read_csv(BASE/'comparison.csv');old['model']=old.model+'_'+old['mode']
    all_metrics=pd.concat([old[['tomb','sensor','horizon_days','model','MAE','RMSE','baseline_MAE','n']],avg],ignore_index=True)
    all_metrics.to_csv(OUT/'comparison_all.csv',index=False)
    pred.to_csv(OUT/'predictions.csv.gz',index=False)
    for h in H:
        table=all_metrics[all_metrics.horizon_days==h].pivot(index='sensor',columns='model',values='MAE')
        base=avg[(avg.horizon_days==h)&(avg.model=='optimized')].set_index('sensor').baseline_MAE
        print('SUMMARY',h,dict(optimized_beats_old_self=int((table.optimized<table.transformer_self).sum()),
            optimized_beats_old_env=int((table.optimized<table.transformer_environment).sum()),
            optimized_beats_lstm_env=int((table.optimized<table.lstm_environment).sum()),
            optimized_beats_persistence=int((table.optimized<base.reindex(table.index)-1e-12).sum()),
            guarded_beats_persistence=int((table.guarded<base.reindex(table.index)-1e-12).sum())),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['init','search','final','summarize'],required=True);p.add_argument('--tomb',type=int)
    p.add_argument('--sensors',nargs='+')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='init':
        save_json(OUT/'policy.json',POLICY)
        import shutil
        shutil.copy2(__file__,OUT/'source_snapshot.py')
    elif args.phase=='search':search(args.tomb,args.sensors)
    elif args.phase=='final':final(args.tomb)
    else:summarize()
