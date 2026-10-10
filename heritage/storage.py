from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from contextlib import contextmanager,closing
from datetime import datetime, timezone

def uid():
    return uuid.uuid4().hex

def now():
    return datetime.now(timezone.utc).isoformat()

def root_dir():
    return Path(os.environ.get("HERITAGE_HOME", str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "HeritageStudio")))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

class Project:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.db = self.path / "metadata.sqlite"
        if not self.db.exists():
            raise ValueError("프로젝트 데이터베이스를 찾을 수 없습니다.")
        with self.connect() as c:
            row = c.execute("SELECT value FROM settings WHERE key='schema'").fetchone()
            if not row or row[0] != '1':
                raise ValueError("지원하지 않는 프로젝트 버전입니다.")

    @classmethod
    def create(cls, name, base=None):
        name = name.strip()
        if not name:
            raise ValueError("프로젝트 이름을 입력하세요.")
        path = Path(base or root_dir() / "projects") / uid()
        path.mkdir(parents=True)
        with closing(sqlite3.connect(path / "metadata.sqlite")) as c:
            c.executescript("""
                CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE records(id TEXT PRIMARY KEY, kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX record_kind ON records(kind,created);
            """)
            c.executemany("INSERT INTO settings VALUES(?,?)", [('schema','1'),('name',name),('created',now())])
            c.commit()
        return cls(path)

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=20)
        c.execute("PRAGMA foreign_keys=ON")
        try:
            with c: yield c
        finally: c.close()

    @property
    def name(self):
        return self.setting('name')

    def setting(self, key, default=None):
        with self.connect() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_setting(self, key, value):
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key,str(value)))

    def add(self, kind, payload, record_id=None):
        ident = record_id or uid()
        with self.connect() as c:
            c.execute("INSERT INTO records VALUES(?,?,?,?)", (ident,kind,now(),json.dumps(payload,ensure_ascii=False,allow_nan=False)))
            if kind == 'dataset':
                c.execute("INSERT OR REPLACE INTO settings VALUES('active_dataset',?)", (ident,))
        return ident

    def update(self, ident, payload):
        with self.connect() as c:
            c.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(payload,ensure_ascii=False,allow_nan=False),ident))

    def list(self, kind):
        with self.connect() as c:
            rows = c.execute("SELECT id,created,payload FROM records WHERE kind=? ORDER BY created DESC,id DESC", (kind,)).fetchall()
        return [dict(json.loads(r[2]),id=r[0],created=r[1]) for r in rows]

    def get(self, ident):
        with self.connect() as c:
            r = c.execute("SELECT kind,created,payload FROM records WHERE id=?", (ident,)).fetchone()
        if not r:
            raise ValueError("저장된 항목을 찾을 수 없습니다.")
        return dict(json.loads(r[2]),id=ident,kind=r[0],created=r[1])

    def file(self, relative):
        p = (self.path / relative).resolve()
        if not p.is_relative_to(self.path):
            raise ValueError("프로젝트 외부 경로는 허용하지 않습니다.")
        return p

    def latest_dataset(self):
        active = self.setting('active_dataset', None)
        if active is not None: return self.get(active) if active else None
        rows = self.list('dataset')
        return rows[0] if rows else None

    def load_data(self, ident=None):
        import pandas as pd
        rec = self.get(ident) if ident else self.latest_dataset()
        if not rec:
            raise ValueError("먼저 데이터를 등록하세요.")
        df = pd.read_csv(self.file(rec['file']),index_col=0,parse_dates=True)
        df.index.name = 'time'
        return df, rec

    def export(self, destination):
        destination = Path(destination).resolve()
        if destination.is_relative_to(self.path):
            raise ValueError("백업 파일은 활성 프로젝트 밖에 저장하세요.")
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            db_copy = Path(td)/'metadata.sqlite'
            with self.connect() as src, closing(sqlite3.connect(db_copy)) as dest:
                src.backup(dest)
            files = {'metadata.sqlite':db_copy}
            for p in self.path.rglob('*'):
                if p.is_file() and p.name != 'metadata.sqlite' and p.suffix not in ['.tmp','.lock'] and 'jobs' not in p.relative_to(self.path).parts:
                    files[p.relative_to(self.path).as_posix()] = p
            manifest = {'schema':1,'name':self.name,'files':{k:sha(v) for k,v in files.items()}}
            temp = destination.with_name(destination.name+'.tmp')
            try:
                with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                    for k,v in files.items(): z.write(v,k)
                    z.writestr('backup_manifest.json',json.dumps(manifest,ensure_ascii=False))
                os.replace(temp,destination)
            finally:
                temp.unlink(missing_ok=True)

    @classmethod
    def restore(cls, archive, base=None):
        base = Path(base or root_dir()/'projects')
        base.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='restore_',dir=base))
        try:
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if len(names) != len(set(names)): raise ValueError("중복 경로가 있는 백업입니다.")
                manifest = json.loads(z.read('backup_manifest.json'))
                if manifest.get('schema') != 1: raise ValueError("지원하지 않는 백업 버전입니다.")
                if sum(i.file_size for i in z.infolist()) > 50*1024**3: raise ValueError("백업이 허용 크기를 초과합니다.")
                if set(names) != set(manifest['files']) | {'backup_manifest.json'}: raise ValueError("백업 목록이 일치하지 않습니다.")
                for name, digest in manifest['files'].items():
                    path = (staging/name).resolve()
                    if not path.is_relative_to(staging.resolve()) or ':' in name or '\\' in name:
                        raise ValueError("잘못된 백업 경로입니다.")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src, path.open('wb') as out: shutil.copyfileobj(src,out)
                    if sha(path) != digest: raise ValueError("백업 파일 검증에 실패했습니다: "+name)
            project = cls(staging)
            with project.connect() as c:
                if c.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError("DB 무결성 오류")
            final = base/uid()
            staging.rename(final)
            return cls(final)
        except Exception:
            shutil.rmtree(staging)
            raise

def projects():
    base = root_dir()/'projects'
    base.mkdir(parents=True,exist_ok=True)
    found=[]
    for p in base.iterdir():
        try: found.append(Project(p))
        except (ValueError,sqlite3.Error): continue
    return sorted(found,key=lambda p:p.setting('created',''),reverse=True)
