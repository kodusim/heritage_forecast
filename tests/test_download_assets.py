import hashlib,json,zipfile
import pytest
from tools.download_assets import extract,restore_aliases

def test_extract_rejects_escape_before_any_write(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('valid.txt','x');f.writestr('../outside.txt','bad')
    with pytest.raises(ValueError):extract(z,tmp_path/'out')
    assert not (tmp_path/'out/valid.txt').exists()
    assert not (tmp_path/'outside.txt').exists()

def test_verified_extract_and_duplicate_alias_restore(tmp_path):
    z=tmp_path/'data.zip';content=b'time,value\n2024-01-01,1\n'
    meta={'files':[{'path':'data/a.csv','sha256':hashlib.sha256(content).hexdigest()}],
        'duplicate_path_aliases':{'original-name/a.csv':'data/a.csv'}}
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('data/a.csv',content);f.writestr('asset-manifest.json',json.dumps(meta))
    out=tmp_path/'out';extract(z,out);restore_aliases(out)
    assert (out/'original-name/a.csv').read_bytes()==content
    with pytest.raises(FileExistsError):extract(z,out)
