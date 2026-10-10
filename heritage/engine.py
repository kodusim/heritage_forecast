from __future__ import annotations
import copy
import hashlib
import json
import random
import time
import platform
import numpy as np
import pandas as pd
from .storage import uid,write_json,read_json
from .data import hourly,window_indices

def metrics(actual,prediction):
    a=np.asarray(actual,dtype=float);p=np.asarray(prediction,dtype=float)
    ok=np.isfinite(a)&np.isfinite(p);a=a[ok];p=p[ok]
    if not len(a): return {'n':0,'MAE':None,'RMSE':None,'R2':None}
    err=a-p;den=((a-a.mean())**2).sum()
    return {'n':len(a),'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt((err**2).mean())),
            'R2':float(1-(err**2).sum()/den) if den>1e-20 else None}

def network(kind,n_features,n_horizons,width=32):
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind=='lstm': self.body=nn.LSTM(n_features,width,batch_first=True)
            else:
                self.proj=nn.Linear(n_features,width)
                self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=.1,batch_first=True),num_layers=1)
                positions=torch.arange(4096).float().unsqueeze(1)
                div=torch.exp(torch.arange(0,width,2).float()*(-np.log(10000.)/width))
                pe=torch.zeros(4096,width);pe[:,0::2]=torch.sin(positions*div);pe[:,1::2]=torch.cos(positions*div)
                self.register_buffer('position',pe.unsqueeze(0))
            self.head=nn.Linear(width,n_horizons)
        def forward(self,x):
            if kind=='lstm': h,_=self.body(x)
            else: h=self.body(self.proj(x)+self.position[:,:x.shape[1]])
            return self.head(h[:,-1])
    return Net()

def validate_config(cfg):
    if not 2<=int(cfg['lookback'])<=720: raise ValueError('입력 길이는 2~720시간입니다.')
    h=cfg['horizons']
    if not h or len(set(h))!=len(h) or min(h)<1 or max(h)>720: raise ValueError('예측시간은 중복 없이 1~720시간으로 지정하세요.')
    if pd.Timestamp(cfg['train_end'])>=pd.Timestamp(cfg['val_end']): raise ValueError('검증 종료는 학습 종료 이후여야 합니다.')
    if not cfg['targets']: raise ValueError('타깃 센서를 선택하세요.')
    if not cfg['modes'] or not set(cfg['modes'])<= {'self','environment'}: raise ValueError('입력 조건을 선택하세요.')
    if 'environment' in cfg['modes'] and not cfg['environment']: raise ValueError('환경 포함 조건에는 환경 센서가 필요합니다.')
    if set(cfg['targets'])&set(cfg['environment']): raise ValueError('타깃을 환경 입력으로 지정할 수 없습니다.')
    if not set(cfg['models'])<={'persistence','linear','lstm','transformer'} or not cfg['models']:raise ValueError('모델을 선택하세요.')
    if not 1<=cfg['epochs']<=500:raise ValueError('학습 횟수는 1~500입니다.')

def train(project,cfg,progress=lambda _:None,cancel=lambda:None):
    validate_config(cfg)
    raw,version=project.load_data(cfg['dataset_id']);frame=hourly(raw,minimum=cfg.get('minimum',4),metadata=version['columns'])
    if frame.index.max()<=pd.Timestamp(cfg['val_end']): raise ValueError('검증 종료 이후의 테스트 자료가 없습니다.')
    run_id=uid();base=project.file('runs/'+run_id);base.mkdir(parents=True)
    cfg=dict(cfg,engine_version='1.0.0',python=platform.python_version())
    write_json(base/'config.json',cfg)
    run={'config':cfg,'status':'running','folder':'runs/'+run_id,'models':[],'skipped':[]}
    project.add('run',run,run_id)
    prediction_rows=0;prediction_temp=base/'predictions.tmp.gz';all_metrics=[];losses=[]
    try:
        for target in cfg['targets']:
            cancel()
            if target not in frame:raise ValueError(f'{target}: 데이터에 없는 센서입니다.')
            # Identical origin sets for all requested conditions; environmental availability cannot bias the comparison.
            union=list(dict.fromkeys([target]+(cfg['environment'] if 'environment' in cfg['modes'] else [])))
            splits=window_indices(frame,target,union,cfg['lookback'],cfg['horizons'],cfg['train_end'],cfg['val_end'])
            if any(len(splits[k])<{'train':32,'validation':8,'test':8}[k] for k in splits):
                reason={k:len(v) for k,v in splits.items()};run['skipped'].append({'sensor':target,'valid_windows':reason});progress(f'{target}: 유효 창 부족으로 제외 {reason}');continue
            compare_hash=hashlib.sha256(np.concatenate([splits[k] for k in ('train','validation','test')]).tobytes()).hexdigest()
            y=frame[target].to_numpy(float);horizons=cfg['horizons'];L=cfg['lookback']
            for mode in cfg['modes']:
                inputs=[target]+(cfg['environment'] if mode=='environment' else [])
                arr=frame[inputs].to_numpy(float)
                train_values=arr[frame.index<=pd.Timestamp(cfg['train_end'])]
                mean=np.nanmean(train_values,axis=0);std=np.nanstd(train_values,axis=0);std[std<1e-12]=1
                if not np.isfinite(mean).all():raise ValueError(f'{target}: 학습 구간에 값이 없는 입력이 있습니다.')
                scaled=((arr-mean)/std).astype(np.float32)
                target_scale=float(std[0])
                for kind in list(dict.fromkeys(['persistence']+cfg['models'])):
                    cancel();progress(f'{target} · {mode} · {kind} 시작')
                    random.seed(cfg['seed']);np.random.seed(cfg['seed'])
                    meta={'sensor':target,'mode':mode,'model':kind,'inputs':inputs,'mean':mean.tolist(),'std':std.tolist(),
                          'target_scale':target_scale,'horizons':horizons,'lookback':L,'minimum':cfg.get('minimum',4),
                          'dataset_id':version['id'],'run_id':run_id,'train_end':cfg['train_end'],'val_end':cfg['val_end'],
                          'columns_meta':{c:version['columns'].get(c,{'unit':'','role':'target'}) for c in inputs},'compare_hash':compare_hash}
                    mid=uid();folder=project.file('models/'+mid);folder.mkdir(parents=True)
                    predictor=None
                    if kind=='linear':
                        # Ridge on last value, window mean and endpoint trend; coefficients saved without pickle.
                        def feature(indices):
                            return np.asarray([np.concatenate((scaled[t],scaled[t-L+1:t+1].mean(0),scaled[t]-scaled[t-L+1])) for t in indices])
                        x=feature(splits['train']);xx=np.c_[np.ones(len(x)),x]
                        dy=np.stack([(y[splits['train']+h]-y[splits['train']])/target_scale for h in horizons],axis=1)
                        penalty=np.eye(xx.shape[1]);penalty[0,0]=0
                        coef=np.linalg.solve(xx.T@xx+penalty,xx.T@dy)
                        np.savez(folder/'linear.npz',coef=coef)
                        predictor=lambda ix: np.c_[np.ones(len(ix)),feature(ix)]@coef
                    elif kind in ('lstm','transformer'):
                        import torch
                        torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
                        torch.manual_seed(cfg['seed'])
                        device='cuda' if cfg.get('device')=='cuda' and torch.cuda.is_available() else 'cpu'
                        if cfg.get('device')=='cuda' and device!='cuda':raise ValueError('CUDA를 사용할 수 없습니다. CPU를 선택하세요.')
                        class Windows(torch.utils.data.Dataset):
                            def __init__(self,indices):self.indices=indices
                            def __len__(self):return len(self.indices)
                            def __getitem__(self,i):
                                t=self.indices[i]
                                return scaled[t-L+1:t+1],np.asarray([(y[t+h]-y[t])/target_scale for h in horizons],dtype=np.float32)
                        model=network(kind,len(inputs),len(horizons)).to(device)
                        opt=torch.optim.Adam(model.parameters(),lr=.001);loss_fn=torch.nn.MSELoss()
                        batch=min(128,max(1,2_000_000//(L*L))) if kind=='transformer' else 128
                        meta['batch_size']=batch
                        loaders={k:torch.utils.data.DataLoader(Windows(ix),batch_size=batch,shuffle=k=='train',num_workers=0) for k,ix in splits.items()}
                        best=float('inf');best_state=None;patience=0
                        for epoch in range(cfg['epochs']):
                            model.train();total=0;n=0
                            for xb,yb in loaders['train']:
                                cancel();xb=xb.to(device);yb=yb.to(device);opt.zero_grad();loss=loss_fn(model(xb),yb)
                                if not torch.isfinite(loss):raise ValueError('학습 손실이 유한하지 않습니다. 입력 범위를 확인하세요.')
                                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(xb);n+=len(xb)
                            model.eval();vt=0;vn=0
                            with torch.no_grad():
                                for xb,yb in loaders['validation']:
                                    cancel();vt+=loss_fn(model(xb.to(device)),yb.to(device)).item()*len(xb);vn+=len(xb)
                            vl=vt/vn;losses.append({'model_id':mid,'sensor':target,'model':kind,'mode':mode,'epoch':epoch+1,'train':total/n,'validation':vl})
                            progress(f'{target} / {kind} / {mode} · {epoch+1}/{cfg["epochs"]} epoch · 검증 손실 {vl:.5g}')
                            if vl<best-1e-10:best=vl;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});patience=0
                            else:patience+=1
                            if patience>=5:break
                        model.load_state_dict(best_state);torch.save(best_state,folder/'weights.pt')
                        meta.update(torch_version=str(torch.__version__),device=device)
                        def predict_ix(ix):
                            out=[];model.eval()
                            with torch.no_grad():
                                for xb,_ in torch.utils.data.DataLoader(Windows(ix),batch_size=batch,num_workers=0):
                                    cancel();out.append(model(xb.to(device)).cpu().numpy())
                            return np.concatenate(out)
                        predictor=predict_ix
                    model_metrics=[]
                    for split in ('validation','test'):
                        ix=splits[split];deltas=np.zeros((len(ix),len(horizons))) if kind=='persistence' else predictor(ix)
                        for j,h in enumerate(horizons):
                            actual=y[ix+h];pred=y[ix]+deltas[:,j]*target_scale
                            m=dict(metrics(actual,pred),model_id=mid,sensor=target,mode=mode,model=kind,horizon=h,split=split,compare_hash=compare_hash)
                            baseline=metrics(actual,y[ix]);m['skill_mae']=1-m['MAE']/baseline['MAE'] if baseline['MAE'] and baseline['MAE']>0 else None
                            all_metrics.append(m);model_metrics.append(m)
                            table=pd.DataFrame({'model_id':mid,'sensor':target,'mode':mode,'model':kind,'split':split,
                                'origin':frame.index[ix],'time':frame.index[ix+h],'horizon':h,'actual':actual,'predicted':pred,'baseline':y[ix]})
                            table.to_csv(prediction_temp,index=False,compression='gzip',mode='a',header=prediction_rows==0)
                            prediction_rows+=len(table)
                    meta['metrics']=model_metrics;meta['folder']='models/'+mid;meta['status']='complete'
                    write_json(folder/'manifest.json',meta);project.add('model',meta,mid);run['models'].append(mid)
                    project.update(run_id,run)
        if not prediction_rows:raise ValueError('학습 가능한 센서가 없습니다. 품질표·기간·입력길이·환경 겹침을 확인하세요.')
        prediction_temp.replace(base/'predictions.csv.gz')
        pd.DataFrame(all_metrics).to_csv(base/'metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(losses).to_csv(base/'loss.csv',index=False)
        run['status']='complete';project.update(run_id,run);return run_id
    except BaseException as e:
        run['status']='cancelled' if isinstance(e,InterruptedError) else 'failed';run['error']=str(e)
        # Partial model artifacts remain auditable but cannot be selected as production models.
        for mid in run['models']:
            m=project.get(mid);m['status']='incomplete_run';project.update(mid,m)
        project.update(run_id,run);raise

def predict(project,model_id,dataset_id=None,cancel=lambda:None,origin=None):
    model=project.get(model_id)
    from .history import affected
    if affected(project,model['dataset_id']):raise ValueError('취소한 데이터로 학습한 모델입니다. 현재 데이터로 다시 학습하세요.')
    if model.get('status')!='complete' or project.get(model['run_id']).get('status')!='complete':raise ValueError('완료된 학습의 모델만 예측할 수 있습니다.')
    raw,version=project.load_data(dataset_id)
    for name,meta in model['columns_meta'].items():
        current=version['columns'].get(name,{})
        if dict(current,aggregation=current.get('aggregation','mean'))!=dict(meta,aggregation=meta.get('aggregation','mean')):raise ValueError(f'{name}: 모델 입력의 역할·단위·집계 방식이 현재 자료와 다릅니다.')
    frame=hourly(raw,minimum=model['minimum'],metadata=version['columns']);latest_origin=frame.index[-1]
    if origin:
        requested=pd.Timestamp(origin)
        if requested not in frame.index:raise ValueError('기준 시각은 데이터 범위 안의 정시여야 합니다.')
        if requested<=pd.Timestamp(model['val_end']):raise ValueError('예측 기준은 모델 검증 종료 이후로 지정하세요.')
        frame=frame.loc[:requested]
    L=model['lookback'];inputs=model['inputs'];window=frame[inputs].iloc[-L:]
    if len(window)!=L or not np.isfinite(window.to_numpy()).all():raise ValueError('기준 시각의 입력기간에 결측이 있거나 시간당 관측 수가 부족합니다. 자료를 보완하거나 유효한 기준 시각을 직접 지정하세요. 기준 시각은 자동으로 과거로 이동하지 않습니다.')
    origin=frame.index[-1];anchor=float(frame[model['sensor']].iloc[-1]);x=((window.to_numpy()-model['mean'])/model['std']).astype(np.float32)
    folder=project.file(model['folder']);kind=model['model'];cancel()
    if kind=='persistence':delta=np.zeros(len(model['horizons']))
    elif kind=='linear':
        with np.load(folder/'linear.npz',allow_pickle=False) as z:coef=z['coef']
        delta=np.r_[1,x[-1],x.mean(0),x[-1]-x[0]]@coef
    else:
        import torch
        net=network(kind,len(inputs),len(model['horizons']))
        net.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True));net.eval()
        with torch.no_grad():delta=net(torch.from_numpy(x[None])).numpy()[0]
    values=anchor+delta*model['target_scale']
    if not np.isfinite(values).all():raise ValueError('유한하지 않은 예측값입니다.')
    ident=uid();rel='predictions/'+ident+'.csv';path=project.file(rel);path.parent.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame({'time':[origin+pd.Timedelta(hours=h) for h in model['horizons']], 'origin':origin,
                       'horizon':model['horizons'],'predicted':values,'sensor':model['sensor'],'model_id':model_id})
    table.to_csv(path,index=False,encoding='utf-8-sig')
    project.add('prediction',{'file':rel,'model_id':model_id,'dataset_id':version['id'],'sensor':model['sensor'],
                             'origin':str(origin),'latest_available_bin':str(latest_origin),'historical_origin':origin<latest_origin,'last_target':str(table.time.max()),'unit':model['columns_meta'][model['sensor']]['unit']},ident)
    return ident

def evaluate_prediction(project,prediction_id,dataset_id=None):
    rec=project.get(prediction_id);model=project.get(rec['model_id']);raw,version=project.load_data(dataset_id)
    sensor=rec['sensor']
    current=version['columns'].get(sensor,{});expected=model['columns_meta'][sensor]
    if dict(current,aggregation=current.get('aggregation','mean'))!=dict(expected,aggregation=expected.get('aggregation','mean')):raise ValueError('실제값의 센서 단위·역할·집계 방식이 모델과 다릅니다.')
    frame=hourly(raw,minimum=model['minimum'],metadata=version['columns']);p=pd.read_csv(project.file(rec['file']),parse_dates=['time'])
    p['actual']=frame[sensor].reindex(pd.DatetimeIndex(p.time)).to_numpy();p['error']=p.predicted-p.actual
    ident=uid();rel='evaluations/'+ident+'.csv';path=project.file(rel);path.parent.mkdir(parents=True,exist_ok=True);p.to_csv(path,index=False)
    project.add('evaluation',{'prediction_id':prediction_id,'truth_dataset_id':version['id'],'file':rel,'metrics':metrics(p.actual,p.predicted)},ident)
    return ident
