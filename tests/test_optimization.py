import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.optimize_transformer import build_inputs,prepare_arrays,PatchTransformer
from research.seed_data import daily_frame

def test_inputs_are_causal_and_ignore_absolute_sensor_offset():
    d=pd.DataFrame({'sensor':np.arange(220,dtype=float)},index=pd.date_range('2020-01-01',periods=220))
    x=build_inputs(d,[100],['sensor'],90,np.array([10.]))
    changed=d.copy();changed.iloc[101:]=1e9
    assert torch.equal(x,build_inputs(changed,[100],['sensor'],90,np.array([10.])))
    assert torch.equal(x,build_inputs(d+10000,[100],['sensor'],90,np.array([10.])))

def test_missing_history_has_explicit_mask_and_finite_values():
    d=pd.DataFrame({'sensor':[np.nan,1.,2.,3.]})
    x=build_inputs(d,[3],['sensor'],30,np.array([1.]))
    assert torch.isfinite(x).all()
    assert int(x[0,:,1].sum())==3
    assert torch.equal(x[0,:27,0],torch.zeros(27))

def test_2024_values_cannot_change_2023_training_or_validation():
    d=daily_frame(1)
    cfg=dict(name='test',days=90,mode='environment',width=16)
    x,y,delta,meta=prepare_arrays(d,'5-D3',cfg,2023)
    changed=d.copy();changed.loc[changed.index>='2024-01-01']=123456789.
    xx,yy,dd,mm=prepare_arrays(changed,'5-D3',cfg,2023)
    # DataFrame copies can change array layout and reduction rounding at float64
    # machine precision. Keep all non-numeric metadata and float32 inputs exact.
    np.testing.assert_allclose(meta['feature_scale'],mm['feature_scale'],rtol=1e-14,atol=0)
    assert {k:v for k,v in meta.items() if k!='feature_scale'}=={k:v for k,v in mm.items() if k!='feature_scale'}
    for split in ['train','validation']:
        assert torch.equal(x[split],xx[split]);assert torch.equal(y[split],yy[split])

def test_initial_network_does_not_invent_motion():
    net=PatchTransformer(3,180,16).eval()
    with torch.no_grad():p=net(torch.randn(4,180,6))
    assert torch.equal(p,torch.zeros(4,2))
