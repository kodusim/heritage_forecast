import pandas as pd
from research import seed_data

def test_seed_join_preserves_benchmark_timestamps_and_optional_environment_only_rows(tmp_path,monkeypatch):
    index=pd.date_range('2024-01-01',periods=3,freq='10min')
    behavior=pd.DataFrame({'5-D1':[1.,3.]},index=index[[0,2]])
    environment=pd.DataFrame({'internal_primary':[10.,20.,30.]},index=index)
    behavior.to_csv(tmp_path/'behavior.csv.gz');environment.to_csv(tmp_path/'environment.csv.gz')
    monkeypatch.setattr(seed_data,'SEED',tmp_path)
    # File ordering in the manifest must not determine the alignment policy.
    monkeypatch.setattr(seed_data,'manifest',lambda:{'files':[
        {'tomb':1,'role':'environment','file':'environment.csv.gz'},
        {'tomb':1,'role':'behavior','file':'behavior.csv.gz'}]})
    left=seed_data.observations(1);outer=seed_data.observations(1,how='outer')
    assert left.index.equals(behavior.index)
    assert len(outer)==3 and pd.isna(outer.loc[index[1],'5-D1'])
    assert outer.loc[index[1],'internal_primary']==20
