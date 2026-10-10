"""Reversible registration removal. Rebuild from preserved sources, then commit atomically."""
import json
import pandas as pd
from .storage import uid, now, sha
from .data import read_sheet, merge_frames, quality


def disabled(project):
    return set(json.loads(project.setting('disabled_imports', '[]')))


def sources(project, ident):
    if not ident:
        return []
    rec = project.get(ident)
    if 'source_ids' in rec:
        return rec['source_ids']
    return sources(project, rec.get('parent')) + [ident]


def affected(project, dataset_id):
    return bool(disabled(project).intersection(sources(project, dataset_id)))


def registrations(project):
    all_rows={r['id']:r for r in project.list('dataset')}
    ordered=[];seen=set()
    def visit(r):
        if r['id'] in seen:return
        seen.add(r['id'])
        if r.get('parent') in all_rows:visit(all_rows[r['parent']])
        for ident in r.get('source_ids',[]):
            if ident in all_rows:visit(all_rows[ident])
        if not r.get('rebuild'):ordered.append(r)
    for r in sorted(all_rows.values(),key=lambda x:(x['created'],x.get('import_order',0),x['id'])):visit(r)
    return ordered


def change_registration(project, ident, enabled, progress=lambda _: None, cancel=lambda: None):
    rows = registrations(project)
    if ident not in {r['id'] for r in rows}:
        raise ValueError('등록 항목을 찾을 수 없습니다.')
    excluded = disabled(project)
    if enabled: excluded.discard(ident)
    else: excluded.add(ident)
    frame = None
    meta = {}
    source_ids = []
    for rec in rows:
        cancel()
        if rec['id'] in excluded: continue
        progress('자료 재구성: ' + rec['source_name'])
        if not rec.get('mappings') or not rec.get('source_file'):
            # Synthetic/legacy base snapshots can be kept as a whole, never split.
            if source_ids or rec.get('parent'):
                raise ValueError('이전 형식의 합본은 개별 등록을 재구성할 수 없습니다. 백업을 복원해 주세요.')
            frame, _ = project.load_data(rec['id']); meta = dict(rec['columns'])
        else:
            path = project.file(rec['source_file'])
            if sha(path) != rec['source_hash']: raise ValueError('보관 원본의 무결성 검사에 실패했습니다.')
            parts = []
            for m in rec['mappings']:
                part, _ = read_sheet(path,m['sheet'],m['header'],m['time_col'],m['columns'],progress,cancel)
                parts.append(part)
                for name in m['columns'].values():
                    desc = rec['columns'][name]
                    if name in meta and dict(meta[name],aggregation=meta[name].get('aggregation','mean')) != dict(desc,aggregation=desc.get('aggregation','mean')):
                        raise ValueError('센서 단위가 다른 등록이 있습니다: ' + name)
                    meta[name] = desc
            frame, _ = merge_frames(frame,pd.concat(parts),rec.get('conflict_policy','reject'))
        source_ids.append(rec['id'])
    cancel()
    rid = uid()
    payload = None
    if frame is not None:
        rel = 'datasets/' + rid + '.csv.gz'
        project.file(rel).parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(project.file(rel),compression='gzip',float_format='%.17g')
        payload = dict(file=rel,parent=None,rebuild=True,source_ids=source_ids,rows=len(frame),
            start=str(frame.index.min()),end=str(frame.index.max()),columns=meta,quality=quality(frame),
            source_name='등록 취소·복원 후 현재 데이터',conflicts=0,invalid_dates=0)
    cancel()
    with project.connect() as c:
        if payload:
            c.execute('INSERT INTO records VALUES(?,?,?,?)',(rid,'dataset',now(),json.dumps(payload,ensure_ascii=False)))
        for key,value in [('active_dataset',rid if payload else ''),('disabled_imports',json.dumps(sorted(excluded)))]:
            c.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,value))
        c.execute('INSERT INTO records VALUES(?,?,?,?)',(uid(),'data_change',now(),json.dumps({'registration':ident,'enabled':enabled,'active_dataset':rid if payload else None})))
    return rid if payload else None


def eligible_origins(project, model):
    import numpy as np
    from .data import hourly
    if affected(project, model['dataset_id']):
        raise ValueError('취소한 자료로 학습한 모델입니다. 현재 데이터로 다시 학습하세요.')
    raw, version = project.load_data()
    for name in model['inputs']:
        current = version['columns'].get(name,{})
        expected = model['columns_meta'][name]
        if dict(current,aggregation=current.get('aggregation','mean')) != dict(expected,aggregation=expected.get('aggregation','mean')):
            raise ValueError('모델과 현재 데이터의 센서·단위가 다릅니다: ' + name)
    frame = hourly(raw,minimum=model['minimum'],metadata=version['columns'])
    ok = pd.Series(np.isfinite(frame[model['inputs']].to_numpy()).all(axis=1),index=frame.index)
    valid = ok.rolling(model['lookback']).sum().eq(model['lookback'])
    valid &= np.isfinite(frame[model['sensor']])
    valid &= frame.index > pd.Timestamp(model['val_end'])
    return frame.index[valid], frame.index[-1]
