import numpy as np
import pandas as pd
import pytest
from heritage.demo import create_demo
from heritage.engine import train,validate_config,predict
from heritage.workflow import preflight,future_curve,forecast_view


def config(project):
    df,d=project.load_data()
    return dict(dataset_id=d['id'],targets=['DEMO-D1'],environment=[],modes=['self'],models=['lstm','transformer'],
        lookback=24,horizons=[24,48,72],forecast_step_hours=24,train_end=str(df.index[850]),val_end=str(df.index[1100]),
        minimum=1,epochs=1,seed=42,device='cpu',learning_rate=.002,width=16,layers=2,dropout=.2,batch_size=32,evaluation_stride=8)


def test_hyperparameters_and_complete_future_path(tmp_path):
    p=create_demo(tmp_path/'projects');cfg=config(p);assert preflight(p,cfg)[0]['ready']
    rid=train(p,cfg)
    for mid in p.get(rid)['models']:
        m=p.get(mid);assert m['width']==16 and m['layers']==2 and m['learning_rate']==.002
        ident=future_curve(p,mid,48);rec,observed,table=forecast_view(p,ident)
        assert table.horizon.tolist()==[24,48]
        assert (pd.to_datetime(table.time)==pd.Timestamp(rec['origin'])+pd.to_timedelta([24,48],unit='h')).all()
        assert observed.index.max()==pd.Timestamp(rec['origin'])
        assert np.isfinite(table.predicted).all()
        # Re-loading configurable architecture must reproduce the saved holdout values.
        backtest=pd.read_csv(p.file(p.get(rid)['folder']+'/predictions.csv.gz'))
        sample=backtest[(backtest.model_id==mid)&(backtest.split=='test')];origin=sample.origin.max()
        replay=p.get(predict(p,mid,origin=origin));actual=pd.read_csv(p.file(replay['file']))
        expected=sample[sample.origin==origin].sort_values('horizon')
        np.testing.assert_allclose(actual.predicted,expected.predicted,rtol=1e-5,atol=1e-7)
        with pytest.raises(ValueError,match='다시 학습'):future_curve(p,mid,96)


def test_preflight_short_history_and_regular_grid(tmp_path):
    p=create_demo(tmp_path/'projects');cfg=config(p);cfg['horizons']=list(range(24,721,24))
    assert not preflight(p,cfg)[0]['ready']
    cfg['horizons']=[24,72]
    with pytest.raises(ValueError,match='일정한 간격'):validate_config(cfg)


def test_latest_incomplete_bin_never_silently_moves_back(tmp_path):
    p=create_demo(tmp_path/'projects');cfg=config(p);cfg['models']=['persistence'];rid=train(p,cfg);mid=p.get(rid)['models'][0]
    df,version=p.load_data();df.loc[df.index[-1],'DEMO-D1']=np.nan
    # Preserve a newer valid sensor value with an input gap in its lookback.
    df.loc[df.index[-3],'DEMO-D1']=np.nan
    rel='datasets/gap.csv.gz';df.to_csv(p.file(rel),compression='gzip');p.add('dataset',dict(version,file=rel))
    before=len(p.list('prediction'))
    with pytest.raises(ValueError,match='결측'):future_curve(p,mid,48)
    assert len(p.list('prediction'))==before
