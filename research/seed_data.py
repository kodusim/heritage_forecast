"""Versioned initial observations and daily aggregation; no external workspace paths."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/seed'

def manifest():return json.loads((SEED/'manifest.json').read_text(encoding='utf-8'))

def verify():
    for item in manifest()['files']:
        path=(SEED/item['file']).resolve()
        if not path.is_relative_to(SEED.resolve()):raise ValueError('Invalid seed path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f"Seed checksum mismatch: {item['id']}")
    return True

def observations(tomb,how='left'):
    entries=[r for r in manifest()['files'] if r['tomb']==tomb]
    if len(entries)!=2:raise ValueError('Expected behavior and environment data')
    frames={r['role']:pd.read_csv(SEED/r['file'],index_col=0,parse_dates=True,float_precision='round_trip') for r in entries}
    return frames['behavior'].join(frames['environment'],how=how)

def daily_frame(tomb):
    raw=observations(tomb)
    groups=raw.resample('D',closed='left',label='right')
    return groups.mean().where(groups.count()>=96)

def prepare():
    verify()
    from .compare_long_horizon import prepare as prepare_experiment
    for tomb in [1,2,3]:prepare_experiment(tomb)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','prepare']);args=p.parse_args()
    if args.command=='verify':verify();print('Seed checksums OK')
    else:prepare()
