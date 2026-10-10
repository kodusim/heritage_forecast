"""Import initial observations into isolated GUI projects, preserving source versions."""
import argparse,os,tempfile
from pathlib import Path
from research.seed_data import verify,observations,manifest
from heritage.storage import Project,projects
from heritage.data import import_data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--home',type=Path);parser.add_argument('--tomb',type=int,choices=[1,2,3])
    args=parser.parse_args()
    if args.home:os.environ['HERITAGE_HOME']=str(args.home.resolve())
    verify()
    for tomb in ([args.tomb] if args.tomb else [1,2,3]):
        key=f'initial_observations_v1_tomb{tomb}'
        if any(p.setting('seed_dataset')==key for p in projects()):print('Already imported:',tomb);continue
        df=observations(tomb,how='outer')
        units={c:u for row in manifest()['files'] if row['tomb']==tomb for c,u in row['units'].items()}
        meta={c:{'role':'environment' if c.startswith('internal_') else 'target','unit':units[c],'aggregation':'mean'} for c in df}
        with tempfile.TemporaryDirectory(prefix='heritage_seed_') as temp:
            path=Path(temp)/f'tomb{tomb}_initial.csv';df.to_csv(path,index_label='time',float_format='%.17g')
            project=Project.create(f'{tomb}번 고분 · 최초 계측 데이터')
            import_data(project,{'path':str(path),'mappings':[{'sheet':'CSV','header':0,'time_col':0,
                'columns':{str(i+1):c for i,c in enumerate(df.columns)}}],'columns_meta':meta,'conflicts':'reject'},print)
            project.set_setting('seed_dataset',key)
        print('Imported:',project.name)

if __name__=='__main__':main()
