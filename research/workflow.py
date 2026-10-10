"""Reproduce baseline and optimized experiments without manually arranging artifacts."""
import argparse,concurrent.futures,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from .seed_data import ROOT,prepare,verify

def execute(module,*args):
    subprocess.run([sys.executable,'-m','research.'+module,*map(str,args)],cwd=ROOT,check=True)

def tombs(module,args,jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(execute,module,*args,'--tomb',t) for t in [1,2,3]]
        for future in futures:future.result()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','baseline','optimize','audit','export'])
    p.add_argument('--jobs',type=int,choices=[1,2,3],default=1);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='baseline':
        verify();tombs('compare_long_horizon',[],a.jobs);execute('compare_long_horizon','--summarize');execute('audit_long_horizon')
    elif a.command=='optimize':
        verify()
        from .optimize_transformer import OUT,POLICY,save_json
        if not (ROOT/'artifacts/long_horizon/comparison.csv').exists():p.error('Run baseline first: python -m research.workflow baseline')
        if (OUT/'policy.json').exists():
            if json.loads((OUT/'policy.json').read_text(encoding='utf-8'))!=POLICY:p.error('Search policy changed; use a separate artifacts directory')
        else:execute('optimize_transformer','--phase','init')
        tombs('optimize_transformer',['--phase','search'],a.jobs)
        files=list((OUT/'sensors').glob('*/selection.json'))
        if len(files)!=22:raise RuntimeError('Incomplete model selection')
        hashes={f.parent.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        freeze=OUT/'selection_freeze.json'
        if freeze.exists():
            if json.loads(freeze.read_text())['selections']!=hashes:raise RuntimeError('Frozen selections changed')
        else:save_json(freeze,dict(created_utc=datetime.now(timezone.utc).isoformat(),selections=hashes))
        tombs('optimize_transformer',['--phase','final'],a.jobs)
        execute('optimize_transformer','--phase','summarize');execute('review_optimization');execute('export_comparison')
    elif a.command=='audit':execute('audit_long_horizon');execute('review_optimization')
    else:execute('export_long_horizon');execute('review_optimization');execute('export_comparison')

if __name__=='__main__':main()
