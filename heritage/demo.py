from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .storage import Project
from .data import import_data

def create_demo(base=None):
    p=Project.create('연습용 · 합성 계측 데이터',base)
    rng=np.random.default_rng(42);n=1440;t=np.arange(n);idx=pd.date_range('2024-01-01',periods=n,freq='h')
    temperature=15+7*np.sin(t/180)+2*np.sin(t/24*2*np.pi)
    y=.003*temperature+.00002*t+rng.normal(0,.0008,n)
    df=pd.DataFrame({'Date':idx,'DEMO-D1':y,'Temperature':temperature})
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'synthetic_demo.csv';df.to_csv(src,index=False)
        import_data(p,{'path':str(src),'mappings':[{'sheet':'CSV','header':0,'time_col':0,'columns':{'1':'DEMO-D1','2':'Temperature'}}],
                    'columns_meta':{'DEMO-D1':{'role':'target','unit':'mm'},'Temperature':{'role':'environment','unit':'°C'}},'conflicts':'reject'})
    p.set_setting('demo','true')
    return p
