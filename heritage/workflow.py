"""User-facing data checks and complete, directly trained forecast paths."""
import pandas as pd
import numpy as np
from .data import hourly,window_indices
from .engine import validate_config,predict
from .history import affected


def preflight(project,cfg,progress=lambda _:None,cancel=lambda:None):
    validate_config(cfg);progress('관측 자료와 학습·평가 기간을 검사하고 있습니다.')
    raw,version=project.load_data(cfg['dataset_id'])
    if affected(project,version['id']):raise ValueError('취소한 자료를 포함한 데이터입니다. 현재 데이터로 다시 설정하세요.')
    frame=hourly(raw,minimum=cfg.get('minimum',4),metadata=version['columns']);rows=[]
    for sensor in cfg['targets']:
        cancel();inputs=list(dict.fromkeys([sensor]+(cfg['environment'] if 'environment' in cfg['modes'] else [])))
        try:
            splits=window_indices(frame,sensor,inputs,cfg['lookback'],cfg['horizons'],cfg['train_end'],cfg['val_end'])
            counts={key:len(value) for key,value in splits.items()}
            ready=counts['train']>=32 and counts['validation']>=8 and counts['test']>=8
            reason='' if ready else '관측 기간을 늘리거나 예측 기간·과거 참고 길이를 줄이세요. 빈 구간도 확인하세요.'
            if ready:
                stride=min(cfg.get('evaluation_stride',1),max(1,min(counts['validation'],counts['test'])//8))
                for key in ('validation','test'):counts[key]=len(splits[key][::stride])
        except ValueError as e:counts={'train':0,'validation':0,'test':0};ready=False;reason=str(e)
        rows.append(dict(sensor=sensor,ready=ready,reason=reason,**counts))
    return rows


def future_curve(project,model_id,hours,cancel=lambda:None):
    model=project.get(model_id);step=model.get('forecast_step_hours');horizons=model['horizons']
    if not step or horizons!=list(range(step,max(horizons)+1,step)):
        raise ValueError('이 모델은 일부 시점만 예측하는 이전 모델입니다. 학습하기에서 기간 전체를 예측하는 모델을 먼저 만드세요.')
    if hours<step or hours>max(horizons) or hours%step:
        raise ValueError(f'이 모델은 {step}시간 간격, 최대 {max(horizons)}시간까지 학습했습니다. 더 긴 기간은 그 기간으로 다시 학습하세요.')
    raw,version=project.load_data();latest=raw[model['sensor']].last_valid_index()
    if latest is None:raise ValueError('선택 센서에 관측값이 없습니다.')
    origin=latest.floor('h')+pd.Timedelta(hours=1)
    # No silent fallback to an older date when the last hourly bin is incomplete.
    return predict(project,model_id,cancel=cancel,origin=str(origin),horizon_limit=hours)


def forecast_view(project,prediction_id):
    rec=project.get(prediction_id);model=project.get(rec['model_id'])
    raw,version=project.load_data(rec['dataset_id'])
    frame=hourly(raw,minimum=model['minimum'],metadata=version['columns'])
    origin=pd.Timestamp(rec['origin']);span=max(72,rec.get('requested_hours',max(model['horizons'])))
    observed=frame.loc[origin-pd.Timedelta(hours=span):origin,rec['sensor']]
    predicted=pd.read_csv(project.file(rec['file']),parse_dates=['time'])
    return rec,observed,predicted
