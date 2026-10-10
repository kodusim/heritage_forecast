import numpy as np
import pandas as pd
from heritage.demo import create_demo
from heritage.engine import train
from heritage.review import Review,make_panels,load_review,overview_figure,export_overview


def test_review_reads_frozen_test_results(tmp_path):
    project=create_demo(tmp_path/'projects');frame,version=project.load_data()
    cfg={'dataset_id':version['id'],'targets':['DEMO-D1'],'environment':['Temperature'],'modes':['self','environment'],
         'models':['transformer'],'lookback':12,'horizons':[1,24],'train_end':str(frame.index[900]),'val_end':str(frame.index[1150]),
         'minimum':1,'epochs':1,'seed':42,'device':'cpu'}
    run=train(project,cfg)
    changed=frame.copy();changed['DEMO-D1']+=100
    changed.to_csv(project.file('datasets/later.csv.gz'),compression='gzip')
    project.add('dataset',dict(version,file='datasets/later.csv.gz'))
    review=load_review(project)
    assert review.run['id']==run and review.version['id']==version['id']
    assert project.latest_dataset()['id']!=version['id'] and review.observed['DEMO-D1'].max()<1
    assert set(review.predictions['split'])=={'test'} and set(review.predictions['model'])=={'transformer'}
    assert review.predictions.time.min()>pd.Timestamp(cfg['val_end'])
    panels=make_panels([review],24,'both',7)
    assert len(panels)==1 and set(panels[0].lines)=={'self','environment'}
    assert panels[0].scores['self']['n']==panels[0].scores['environment']['n']
    output=tmp_path/'review.png';export_overview(panels,24,output)
    assert output.exists() and output.with_name('review_figure.json').exists()


def test_review_keeps_unavailable_sensors_and_gaps():
    time=pd.date_range('2024-10-01',periods=8,freq='h')
    observed=pd.DataFrame({'x':[1.,1.,np.nan,1.,1.,1.,1.,1.],'missing':np.nan},index=time)
    version={'columns':{'x':{'role':'target','unit':'mm'},'missing':{'role':'target','unit':'°'}}}
    predictions=pd.DataFrame({'sensor':['x','x'],'mode':['environment']*2,'horizon':[1]*2,'time':[time[1],time[5]],'actual':[1.,1.],'predicted':[1.1,1.2],'baseline':[1.,1.]})
    review=Review(None,'test',version,{'config':{'val_end':str(time[0])}},observed,predictions)
    panels=make_panels([review],1)
    assert len(panels)==2 and not panels[1].lines and panels[1].reason
    assert panels[0].lines['environment'].isna().sum()==5
    assert panels[0].scores['environment']['n']==2 and panels[0].scores['environment']['improvement_pct'] is None
    figure=overview_figure(panels,1,screen=True)
    assert len(figure.axes)==2 and np.isnan(figure.axes[0].lines[1].get_ydata()).any()
