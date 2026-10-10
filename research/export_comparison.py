"""Export matched LSTM vs optimized Transformer MAE, RMSE, and level R-squared."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def export():
    base=ROOT/'artifacts/long_horizon';out=ROOT/'artifacts/optimized_transformer'
    a=pd.read_csv(base/'predictions_all.csv.gz')
    a=a[(a.model=='lstm')&(a['mode']=='environment')].copy();a['model']='LSTM';a['pred']=a.prediction
    b=pd.read_csv(out/'predictions.csv.gz');b['model']='Transformer';b['pred']=b.optimized
    keys=['sensor','seed','horizon_days','origin_time','target_time']
    matched=a.merge(b,on=keys,suffixes=('_l','_t'),validate='one_to_one')
    if len(matched)!=len(a) or len(matched)!=len(b) or not np.allclose(matched.actual_l,matched.actual_t):
        raise ValueError('Models do not use the same evaluation observations')
    rows=[]
    for frame in [a,b]:
        for (sensor,horizon,seed),g in frame.groupby(['sensor','horizon_days','seed']):
            err=g.pred.to_numpy()-g.actual.to_numpy();den=((g.actual-g.actual.mean())**2).sum()
            rows.append(dict(sensor=sensor,horizon_days=horizon,seed=seed,model=g.model.iloc[0],n=len(g),
                MAE=np.abs(err).mean(),RMSE=np.sqrt(np.mean(err**2)),R2=1-np.sum(err**2)/den if den>0 else np.nan))
    result=pd.DataFrame(rows).groupby(['sensor','horizon_days','model']).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),R2=('R2','mean'),n=('n','first')).reset_index()
    result.to_csv(out/'LSTM_vs_optimized_Transformer_R2.csv',index=False,encoding='utf-8-sig')
    return result

if __name__=='__main__':export()
