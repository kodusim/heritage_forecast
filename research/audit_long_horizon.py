"""Independent checks on completed experiment artifacts and forecast causality."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
from .compare_long_horizon import OUT,ROOT
from .models import LSTMModel,TransformerModel

def audit(partial=False):
    torch.set_num_threads(1)
    files=list((OUT/'runs').glob('*/predictions.csv'))
    if not partial: assert len(files)==264, (len(files),264)
    origins={};checks=0;reproduced=0
    daily={t:pd.read_csv(OUT/f'tomb{t}_daily.csv',index_col=0,parse_dates=True) for t in [1,2,3]}
    for path in files:
        p=pd.read_csv(path,parse_dates=['origin_time','target_time'])
        first=p.iloc[0];d=daily[int(first.tomb)]
        meta=json.loads((path.parent/'metadata.json').read_text())
        assert len(p)==2*len(meta['test_indices'])
        assert (p.target_time-p.origin_time).dt.days.eq(p.horizon_days).all()
        assert (p.origin_time>=pd.Timestamp('2024-01-01')).all()
        assert np.allclose(p.actual,d.loc[p.target_time,first.sensor].to_numpy())
        assert np.allclose(p.origin,d.loc[p.origin_time,first.sensor].to_numpy())
        assert np.allclose(p.persistence,p.origin)
        valid=d[[first.sensor,'internal_primary','internal_backup']].notna().all(axis=1).rolling(30).sum().eq(30).to_numpy()
        for split in ['train','validation','test']:
            ix=np.array(meta[split+'_indices']); last=d.index[ix+30]
            if split=='train': assert (last<pd.Timestamp('2023-01-01')).all()
            if split=='validation':
                assert (d.index[ix]>=pd.Timestamp('2023-01-01')).all()
                assert (last<pd.Timestamp('2024-01-01')).all()
            assert valid[ix].all()
            assert d[first.sensor].iloc[ix+7].notna().all()
            assert d[first.sensor].iloc[ix+30].notna().all()
        key=(int(first.tomb),first.sensor)
        stamps=p.origin_time.to_numpy()
        if key in origins: assert np.array_equal(origins[key],stamps)
        else: origins[key]=stamps
        arr=d[meta['inputs']].to_numpy()
        assert np.allclose(meta['mean'],np.nanmean(arr[d.index<'2023-01-01'],axis=0))
        # Reproduce a forecast, then change ALL future data. Output must be identical.
        if first.seed==42:
            cls=LSTMModel if first.model=='lstm' else TransformerModel
            net=cls(len(meta['inputs']),2)
            net.load_state_dict(torch.load(path.parent/'weights.pt',map_location='cpu',weights_only=True));net.eval()
            t=meta['test_indices'][0]
            def predict(a):
                x=(a[t-29:t+1]-meta['mean'])/meta['std']
                with torch.no_grad(): delta=net(torch.tensor(x[None],dtype=torch.float32)).numpy()[0]
                return a[t,0]+delta*np.array(meta['delta_scale'])
            before=predict(arr)
            changed=arr.copy();changed[t+1:]=1234567
            assert np.array_equal(before,predict(changed))
            expected=p[p.origin_time==d.index[t]].sort_values('horizon_days').prediction.to_numpy()
            assert np.allclose(before,expected,rtol=1e-6,atol=1e-8)
            reproduced+=1
        checks+=1
    result=dict(runs=checks,sensors=len(origins),forecast_reproduction_and_future_perturbation_checks=reproduced,
        identical_origins=True,disjoint_training_validation_test_targets=True,
        observed_targets_only=True,future_inputs_excluded=True)
    (OUT/('audit_partial.json' if partial else 'audit.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':audit('--partial' in sys.argv)
