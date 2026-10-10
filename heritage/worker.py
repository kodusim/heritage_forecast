import sys
import traceback
from pathlib import Path
from .storage import Project,read_json,write_json

def main(request_path):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    path=Path(request_path);request=read_json(path);project=Project(request['project'])
    def progress(message):
        import json
        line=json.dumps({'message':message},ensure_ascii=False)
        with path.with_suffix('.events.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
        if sys.stdout is not None:print(line,flush=True)
    def cancel():
        if path.with_suffix('.cancel').exists():raise InterruptedError('사용자가 작업을 취소했습니다.')
    try:
        kind=request['kind'];cfg=request['config'];cancel()
        if kind=='import':
            from .data import import_data
            result=import_data(project,cfg,progress,cancel)
        elif kind=='registration':
            from .history import change_registration
            result=change_registration(project,cfg['id'],cfg['enabled'],progress,cancel)
        elif kind=='origins':
            from .history import eligible_origins
            valid,latest=eligible_origins(project,project.get(cfg['model_id']))
            result={'times':[str(x) for x in valid],'latest':str(latest),'model_id':cfg['model_id']}
        elif kind=='preflight':
            from .workflow import preflight
            result=preflight(project,cfg,progress,cancel)
        elif kind=='future_curve':
            from .workflow import future_curve
            result=future_curve(project,cfg['model_id'],cfg['hours'],cancel)
        elif kind=='train':
            from .engine import train
            result=train(project,cfg,progress,cancel)
        elif kind=='predict':
            from .engine import predict
            result=predict(project,cfg['model_id'],cfg.get('dataset_id'),cancel,origin=cfg.get('origin'))
        elif kind=='evaluate':
            from .engine import evaluate_prediction
            result=evaluate_prediction(project,cfg['prediction_id'],cfg.get('dataset_id'))
        elif kind=='backup':project.export(cfg['destination']);result=cfg['destination']
        else:raise ValueError('알 수 없는 작업입니다.')
        write_json(path.with_suffix('.result.json'),{'ok':True,'id':result});progress('작업 완료')
        return 0
    except BaseException as e:
        write_json(path.with_suffix('.result.json'),{'ok':False,'error':str(e),'cancelled':isinstance(e,InterruptedError)})
        path.with_suffix('.error.log').write_text(traceback.format_exc(),encoding='utf-8');progress(str(e));return 1
