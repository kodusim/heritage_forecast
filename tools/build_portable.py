"""Build a source distribution with the official embedded CPython runtime.

No system Python installation, PATH changes or security-policy changes are required.
Run this script with the development virtual environment after installing requirements.
"""
import hashlib
import importlib.metadata as metadata
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'portable'/'HeritageStudio'
RUNTIME=DEST/'runtime'
PACKAGES=DEST/'packages'
APP=DEST/'app'
for path in (RUNTIME,PACKAGES,APP):path.mkdir(parents=True,exist_ok=True)
archive=ROOT/'.cache'/'python-3.12.10-embed-amd64.zip'
archive.parent.mkdir(parents=True,exist_ok=True)
url='https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip'
if not archive.exists():urllib.request.urlretrieve(url,archive)
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        if not (RUNTIME/member.filename).resolve().is_relative_to(RUNTIME.resolve()):raise ValueError('Invalid runtime archive path')
    z.extractall(RUNTIME)
(RUNTIME/'python312._pth').write_text('python312.zip\n.\n../app\n../packages\nimport site\n',encoding='utf-8')
pending=[Requirement(line).name for line in (ROOT/'requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
seen=set();versions={};copied=0
while pending:
    name=pending.pop();normalized=name.lower().replace('_','-')
    if normalized in seen:continue
    seen.add(normalized);dist=metadata.distribution(name);versions[dist.metadata['Name']]=dist.version
    print('Bundling',name,dist.version,flush=True)
    for req in dist.requires or []:
        item=Requirement(req)
        if not item.marker or item.marker.evaluate({'extra':''}):pending.append(item.name)
    for file in dist.files or []:
        relative=Path(file)
        if '..' in relative.parts or '__pycache__' in relative.parts or relative.is_absolute():continue
        source=Path(dist.locate_file(file));target=PACKAGES/relative
        if source.is_file():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copied+=1
shutil.copy2(ROOT/'main.py',APP/'main.py')
shutil.copytree(ROOT/'heritage',APP/'heritage',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('README','NOTICE'):
    if (ROOT/name).exists():shutil.copy2(ROOT/name,DEST/name)
(DEST/'run.cmd').write_text('@echo off\ncd /d "%~dp0"\nstart "" "runtime\\pythonw.exe" "app\\main.py"\n',encoding='ascii')
(DEST/'diagnose.cmd').write_text('@echo off\ncd /d "%~dp0"\n"runtime\\python.exe" "app\\main.py"\npause\n',encoding='ascii')
(DEST/'distribution.json').write_text(json.dumps({'runtime_source':url,'runtime_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'packages':versions,'package_files':copied},indent=2),encoding='utf-8')
print('Portable distribution ready:',DEST,flush=True)
