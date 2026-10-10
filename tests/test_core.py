import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from heritage.data import dates,merge_frames,window_indices,hourly,import_data,inspect_file,infer_mapping,read_sheet
from heritage.storage import Project,read_json
from heritage.engine import metrics,train,predict,evaluate_prediction
from heritage.demo import create_demo
from heritage.figures import FigureSpec,export_figure

def test_korean_and_excel_dates():
    d=dates(pd.Series(['2025-05-20 오후 12:55:24','2025-05-20 오전 12:55:24',45292.,'junk']))
    assert d.iloc[0].hour==12 and d.iloc[1].hour==0
    assert d.iloc[2]==pd.Timestamp('2024-01-01') and pd.isna(d.iloc[3])

def test_duplicates_conflicts_and_missing():
    idx=pd.to_datetime(['2024-01-01','2024-01-02']);a=pd.DataFrame({'x':[1.,np.nan]},index=idx);b=pd.DataFrame({'x':[2.,3.]},index=idx)
    with pytest.raises(ValueError,match='서로 다른'):merge_frames(a,b)
    assert merge_frames(a,b,'old')[0].x.tolist()==[1.,3.]
    assert merge_frames(a,b,'new')[0].x.tolist()==[2.,3.]
    assert pd.isna(merge_frames(a,b,'mask')[0].x.iloc[0])
    assert len(merge_frames(a,a)[0])==2

def test_no_future_anchor_or_imputation():
    idx=pd.date_range('2024-01-01',periods=240,freq='h');df=pd.DataFrame({'x':np.arange(240.)},index=idx)
    h=[1,6,24];s=window_indices(df,'x',['x'],12,h,idx[120],idx[180])
    assert (idx[s['train']+24]<=idx[120]).all()
    assert (idx[s['validation']+1]>idx[120]).all() and (idx[s['validation']+24]<=idx[180]).all()
    assert (idx[s['test']+1]>idx[180]).all()
    t=s['test'][0];assert df.x.iloc[t+24]-df.x.iloc[t]==24
    altered=df.copy();altered.loc[idx[t+1]:,'x']=999
    np.testing.assert_array_equal(df.x.iloc[t-11:t+1],altered.x.iloc[t-11:t+1])
    df.loc[idx[190:220],'x']=np.nan;s=window_indices(df,'x',['x'],12,h,idx[120],idx[180]);assert all(np.isfinite(df.x.iloc[t-11:t+1]).all() for t in s['test'])

def test_hourly_observation_threshold():
    df=pd.DataFrame({'x':[1,np.nan,3,4,5,6]},index=pd.date_range('2024-01-01',periods=6,freq='10min'))
    assert hourly(df,minimum=6).x.isna().all()
    h=hourly(df,minimum=4);assert h.index[0]==pd.Timestamp('2024-01-01 01:00')

def test_model_roundtrip_and_figures(tmp_path):
    p=create_demo(tmp_path/'projects');d=p.latest_dataset();df,_=p.load_data()
    cfg={'dataset_id':d['id'],'targets':['DEMO-D1'],'environment':['Temperature'],'modes':['self','environment'],
         'models':['persistence','linear','lstm','transformer'],'lookback':12,'horizons':[1,6,24],
         'train_end':str(df.index[900]),'val_end':str(df.index[1150]),'minimum':1,'epochs':1,'seed':42,'device':'cpu'}
    rid=train(p,cfg);run=p.get(rid);assert run['status']=='complete' and len(run['models'])==8
    met=pd.read_csv(p.file(run['folder']+'/metrics.csv'));assert met.groupby(['split','horizon']).n.nunique().eq(1).all()
    assert met.compare_hash.nunique()==1
    historical=pd.read_csv(p.file(run['folder']+'/predictions.csv.gz'))
    for mid in run['models']:
        rec=p.get(mid);pred_id=predict(Project(p.path),mid);pr=p.get(pred_id)
        q=pd.read_csv(p.file(pr['file']));assert len(q)==3 and np.isfinite(q.predicted).all()
        ev=p.get(evaluate_prediction(p,pred_id));assert ev['metrics']['n']==0
        samples=historical[(historical.model_id==mid)&(historical.split=='test')];origin=samples.origin.max()
        saved=samples[samples.origin==origin].sort_values('horizon')
        replay=p.get(predict(p,mid,origin=origin));reproduced=pd.read_csv(p.file(replay['file'])).sort_values('horizon')
        np.testing.assert_allclose(reproduced.predicted,saved.predicted,rtol=1e-5,atol=1e-7)
    path=tmp_path/'paper.svg';export_figure(df[['DEMO-D1']].iloc[:100],FigureSpec(),path,{'run_id':rid})
    assert path.exists() and path.with_name('paper_figure.json').exists()
    backup=tmp_path/'project.heritage';p.export(backup);restored=Project.restore(backup,tmp_path/'restored')
    assert restored.name==p.name and len(restored.list('model'))==8
    assert predict(restored,run['models'][0])

def test_cancel_never_completes_run(tmp_path):
    p=create_demo(tmp_path/'projects');df,d=p.load_data()
    cfg={'dataset_id':d['id'],'targets':['DEMO-D1'],'environment':[],'modes':['self'],'models':['lstm'],
         'lookback':12,'horizons':[1], 'train_end':str(df.index[900]),'val_end':str(df.index[1150]),'minimum':1,'epochs':3,'seed':42,'device':'cpu'}
    def cancelled():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):train(p,cfg,cancel=cancelled)
    assert p.list('run')[0]['status']=='cancelled' and not p.list('model')

def test_archive_traversal(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('../outside.txt','bad');f.writestr('backup_manifest.json',json.dumps({'schema':1,'files':{'../outside.txt':'bad'}}))
    with pytest.raises(ValueError):Project.restore(z,tmp_path/'restore')
    assert not (tmp_path/'outside.txt').exists()

def test_import_repeated_and_units(tmp_path):
    p=Project.create('test',tmp_path/'projects');path=tmp_path/'data.csv'
    pd.DataFrame({'date':pd.date_range('2024-01-01',periods=10,freq='h'),'x':range(10)}).to_csv(path,index=False)
    req={'path':str(path),'mappings':[{'sheet':'CSV','header':0,'time_col':0,'columns':{'1':'x'}}], 'columns_meta':{'x':{'role':'target','unit':'mm'}},'conflicts':'reject'}
    import_data(p,req)
    with pytest.raises(ValueError,match='이미 등록'):import_data(p,req)
    req['columns_meta']['x']['unit']='m'
    with pytest.raises(ValueError,match='단위'):import_data(p,req)

def test_xlsx_headers_and_cached_values(tmp_path):
    from openpyxl import Workbook
    p=tmp_path/'data.xlsx';w=Workbook();s=w.active;s.title='2024';s.append(['대상',None,None]);s.append(['날짜','5-D1','5-T1'])
    for i,t in enumerate(pd.date_range('2024-01-01',periods=20,freq='10min')):s.append([t.to_pydatetime(),i,i/10])
    w.save(p);info=inspect_file(p);h,d=infer_mapping(info[0]['rows']);assert h==1 and d==0
    frame,n=read_sheet(p,'2024',h,d,{'1':'5-D1','2':'5-T1'});assert len(frame)==20 and frame['5-D1'].iloc[-1]==19

def test_constant_metric():
    m=metrics([1,1],[1,2]);assert m['R2'] is None and m['MAE']==.5


def test_physical_aggregation():
    d=pd.DataFrame({'wind':[350.,10.],'rain':[1.,2.],'cumulative':[3.,5.]},index=pd.date_range('2024-01-01',periods=2,freq='20min'))
    h=hourly(d,metadata={'wind':{'aggregation':'circular'},'rain':{'aggregation':'sum'},'cumulative':{'aggregation':'last'}})
    assert min(abs(h.wind.iloc[0]),abs(h.wind.iloc[0]-360))<1e-8
    assert h.rain.iloc[0]==3 and h.cumulative.iloc[0]==5
    d.wind=[0.,180.];assert pd.isna(hourly(d,metadata={'wind':{'aggregation':'circular'}}).wind.iloc[0])


def test_corrupt_archive(tmp_path):
    p=create_demo(tmp_path/'projects');good=tmp_path/'good.heritage';bad=tmp_path/'bad.heritage';p.export(good)
    with zipfile.ZipFile(good) as source,zipfile.ZipFile(bad,'w') as dest:
        for name in source.namelist():dest.writestr(name,b'corrupt' if name.endswith('.csv.gz') else source.read(name))
    with pytest.raises(ValueError,match='검증'):Project.restore(bad,tmp_path/'restored')
    assert not list((tmp_path/'restored').iterdir())


def test_missing_recent_inputs_and_late_actuals(tmp_path):
    p=create_demo(tmp_path/'projects');df,d=p.load_data()
    cfg={'dataset_id':d['id'],'targets':['DEMO-D1'],'environment':[],'modes':['self'],'models':['persistence'],
         'lookback':12,'horizons':[1,6,24],'train_end':str(df.index[900]),'val_end':str(df.index[1150]),'minimum':1,'epochs':1,'seed':42}
    run=p.get(train(p,cfg));mid=run['models'][0];pred=predict(p,mid);forecast=pd.read_csv(p.file(p.get(pred)['file']))
    future=pd.DataFrame({'Date':pd.date_range(df.index[-1]+pd.Timedelta(hours=1),periods=24,freq='h'),'DEMO-D1':.3})
    src=tmp_path/'future.csv';future.to_csv(src,index=False)
    req={'path':str(src),'mappings':[{'sheet':'CSV','header':0,'time_col':0,'columns':{'1':'DEMO-D1'}}], 'columns_meta':{'DEMO-D1':d['columns']['DEMO-D1']},'conflicts':'reject'}
    import_data(p,req);ev=p.get(evaluate_prediction(p,pred));assert ev['metrics']['n']==3
    pd.testing.assert_frame_equal(forecast,pd.read_csv(p.file(p.get(pred)['file'])))
    # An environmental-only newer timestamp must not silently move the prediction origin back.
    future=pd.DataFrame({'Date':[df.index[-1]+pd.Timedelta(hours=25)],'Temperature':[20.]});future.to_csv(src,index=False)
    req['mappings'][0]['columns']={'1':'Temperature'};req['columns_meta']={'Temperature':d['columns']['Temperature']};import_data(p,req)
    with pytest.raises(ValueError,match='결측'):predict(p,mid)
    broken=p.latest_dataset();broken['columns']['DEMO-D1']['unit']='m';p.update(broken['id'],broken)
    with pytest.raises(ValueError,match='단위'):predict(p,mid)


def test_figure_size_and_no_invented_observations(tmp_path):
    from heritage.figures import make_figure
    df=pd.DataFrame({'x':[1.,np.nan,2.]},index=pd.date_range('2024-01-01',periods=3))
    fig=make_figure(df,FigureSpec());assert np.isnan(fig.axes[0].lines[0].get_ydata()[1])
    with pytest.raises(ValueError,match='너무 큽니다'):export_figure(df,FigureSpec(width_mm=400,height_mm=400,dpi=1200),tmp_path/'big.png',{})
    with pytest.raises(ValueError,match='유효'):make_figure(df*np.nan,FigureSpec())


def test_comparison_and_residual_break_missing_hours():
    from heritage.figures import make_figure
    data=pd.DataFrame({'time':['2024-01-01 00:00','2024-01-01 03:00'],'actual':[1.,2.],'predicted':[1.1,2.1],'model':['lstm','lstm'],'mode':['self','self']})
    for kind in ('comparison','residual'):
        fig=make_figure(data,FigureSpec(kind=kind));line=fig.axes[0].lines[0]
        assert len(line.get_ydata())==4 and np.isnan(line.get_ydata()[1:3]).all()


def test_mac_excel_epoch(tmp_path):
    from openpyxl import Workbook
    from openpyxl.utils.datetime import CALENDAR_MAC_1904
    p=tmp_path/'mac.xlsx';w=Workbook();w.epoch=CALENDAR_MAC_1904;s=w.active;s.append(['Date','x']);s.append([pd.Timestamp('2024-01-01').to_pydatetime(),2]);w.save(p)
    df,_=read_sheet(p,s.title,0,0,{'1':'x'});assert df.index[0]==pd.Timestamp('2024-01-01')


def test_csv_encoding_and_compact_dates(tmp_path):
    from heritage.data import csv_encoding
    p=tmp_path/'utf16.csv';p.write_text('Date,x\n20240101,3\n',encoding='utf-16')
    df,_=read_sheet(p,'CSV',0,0,{'1':'x'});assert df.index[0]==pd.Timestamp('2024-01-01')
    p=tmp_path/'utf8.csv';p.write_bytes(b'a'*65535+'한글'.encode('utf-8'))
    assert csv_encoding(p)=='utf-8-sig'
