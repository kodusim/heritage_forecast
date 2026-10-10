"""Download hash-verified release data and restore archives without overwriting files."""
import argparse,hashlib,json,shutil,stat,urllib.request,zipfile,os
from datetime import datetime
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
TAG='development-snapshot-20261011'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def destination(base,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path')
    dest=(base/Path(*rel.parts)).resolve()
    if not dest.is_relative_to(base.resolve()):raise ValueError('Archive path leaves destination')
    return dest

def extract(path,base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        records=[]
        if 'asset-manifest.json' in names:records=json.loads(z.read('asset-manifest.json'))['files']
        # Check every path before extracting anything. Refuse symlinks and existing files.
        for item in z.infolist():
            target=destination(base,item.filename)
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Archive symlink refused')
            if target.exists() and not item.is_dir():raise FileExistsError(target)
        for item in z.infolist():
            target=destination(base,item.filename)
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            stamp=datetime(*item.date_time).timestamp()
            os.utime(target,(stamp,stamp))
        for record in records:
            if digest(destination(base,record['path']))!=record['sha256']:raise ValueError('Extracted file checksum mismatch')

def restore_aliases(base):
    meta=json.loads((base/'asset-manifest.json').read_text(encoding='utf-8'))
    for alias,source in meta.get('duplicate_path_aliases',{}).items():
        src=destination(base,source);dst=destination(base,alias)
        if dst.exists():
            if digest(src)!=digest(dst):raise FileExistsError(dst)
            continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def restore_research(base):
    for old,new in [('long_horizon_results','long_horizon'),('optimized_transformer_results','optimized_transformer')]:
        source=base/'program'/old;target=ROOT/'artifacts'/new
        if not source.is_dir():raise FileNotFoundError(source)
        if target.exists():raise FileExistsError(f'{target}: preserve or move your existing experiment first')
    for old,new in [('long_horizon_results','long_horizon'),('optimized_transformer_results','optimized_transformer')]:
        shutil.copytree(base/'program'/old,ROOT/'artifacts'/new)

def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',default='all',help='Asset filename or stem; default: all')
    p.add_argument('--extract',action='store_true');p.add_argument('--restore-aliases',action='store_true')
    p.add_argument('--restore-research',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'data/releases'/f'{TAG}.json').read_text(encoding='utf-8'))
    assets=[r for r in manifest['assets'] if a.asset in ('all',r['name'],Path(r['name']).stem)]
    if not assets:p.error('Unknown asset name')
    if a.restore_research and not any(r['name']=='trained-models-and-results.zip' for r in assets):p.error('Select trained-models-and-results to restore research')
    for asset in assets:
        cache=ROOT/'.cache/releases'/TAG/asset['name'];cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists():
            url=f"https://github.com/{manifest['repository']}/releases/download/{TAG}/{asset['name']}"
            tmp=cache.with_suffix('.download')
            print('Downloading',asset['name'],flush=True)
            urllib.request.urlretrieve(url,tmp)
            if digest(tmp)!=asset['sha256']:raise ValueError('Download checksum mismatch')
            tmp.replace(cache)
        if digest(cache)!=asset['sha256']:raise ValueError('Cached asset checksum mismatch')
        print('Verified',cache,flush=True)
        if a.extract or a.restore_research or a.restore_aliases:
            dest=ROOT/'artifacts/snapshots'/TAG/cache.stem
            extract(cache,dest)
            if a.restore_aliases and asset['name']=='original-data.zip':restore_aliases(dest)
            if a.restore_research and asset['name']=='trained-models-and-results.zip':restore_research(dest)

if __name__=='__main__':main()
