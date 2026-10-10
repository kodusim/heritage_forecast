import pandas as pd
import numpy as np
import pytest
from heritage.storage import Project
from heritage.data import import_data
from heritage.history import change_registration, affected, eligible_origins, registrations
from heritage.demo import create_demo
from heritage.engine import train, predict


def register(p,path,values,policy='new'):
    pd.DataFrame({'date':pd.date_range('2024-01-01',periods=len(values),freq='h'),'x':values}).to_csv(path,index=False)
    return import_data(p,{'path':str(path),'mappings':[{'sheet':'CSV','header':0,'time_col':0,'columns':{'1':'x'}}],
        'columns_meta':{'x':{'role':'target','unit':'mm'}},'conflicts':policy})


def test_remove_middle_restore_original_values_and_backup(tmp_path):
    p=Project.create('test',tmp_path/'projects')
    a=register(p,tmp_path/'a.csv',[1.,2.,3.]);b=register(p,tmp_path/'b.csv',[9.,8.,7.]);c=register(p,tmp_path/'c.csv',[9.,8.,7.,4.],'old')
    assert [r['id'] for r in registrations(p)]==[a,b,c]
    change_registration(p,b,False)
    assert p.load_data()[0].x.tolist()==[1,2,3,4]
    assert affected(p,c) and not affected(p,p.latest_dataset()['id'])
    change_registration(p,b,True);assert p.load_data()[0].x.tolist()==[9,8,7,4]
    change_registration(p,a,False);change_registration(p,b,False);change_registration(p,c,False)
    assert p.latest_dataset() is None
    p.export(tmp_path/'backup.heritage');q=Project.restore(tmp_path/'backup.heritage',tmp_path/'restored')
    assert q.latest_dataset() is None
    change_registration(q,a,True);assert q.load_data()[0].x.tolist()==[1,2,3]
    d=register(q,tmp_path/'d.csv',[3.,4.,5.]);assert q.latest_dataset()['id']==d


def test_rebuild_failure_is_atomic(tmp_path):
    p=Project.create('atomic',tmp_path/'projects');a=register(p,tmp_path/'a.csv',[1.,2.]);b=register(p,tmp_path/'b.csv',[3.,4.])
    current=p.latest_dataset()['id'];p.file(p.get(a)['source_file']).write_text('corrupt')
    with pytest.raises(ValueError,match='무결성'):change_registration(p,b,False)
    assert p.latest_dataset()['id']==current and not affected(p,b)


def test_cancel_rebuild_keeps_active_version(tmp_path):
    p=Project.create('cancel',tmp_path/'projects');a=register(p,tmp_path/'a.csv',[1.,2.]);b=register(p,tmp_path/'b.csv',[3.,4.])
    def cancel():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):change_registration(p,b,False,cancel=cancel)
    assert p.latest_dataset()['id']==b and not affected(p,b)


def test_valid_origins_exactly_match_engine_and_cancelled_model(tmp_path):
    p=create_demo(tmp_path/'projects');df,d=p.load_data()
    rid=train(p,{'dataset_id':d['id'],'targets':['DEMO-D1'],'environment':[],'modes':['self'],'models':['persistence'],
        'lookback':12,'horizons':[1,6,24],'train_end':str(df.index[900]),'val_end':str(df.index[1150]),'minimum':1,'epochs':1,'seed':42,'device':'cpu'})
    m=p.get(p.get(rid)['models'][0]);times,latest=eligible_origins(p,m)
    assert latest in times and (times>pd.Timestamp(m['val_end'])).all()
    for t in (times[0],times[-1]):assert predict(p,m['id'],origin=str(t))
    change_registration(p,d['id'],False)
    with pytest.raises(ValueError,match='취소'):predict(p,m['id'])
    change_registration(p,d['id'],True);assert predict(p,m['id'])
