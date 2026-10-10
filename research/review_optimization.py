"""Audit and export optimized Transformer artifacts after selection is frozen."""
import json,hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .optimize_transformer import BASE,OUT,H,POLICY,load_json,save_json,PatchTransformer,build_inputs

def audit():
    torch.set_num_threads(1)
    files=list((OUT/'sensors').glob('*/final/*/predictions.csv'));assert len(files)==66
    freeze=load_json(OUT/'selection_freeze.json');assert len(freeze['selections'])==22
    for sensor,digest in freeze['selections'].items():
        assert hashlib.sha256((OUT/'sensors'/sensor/'selection.json').read_bytes()).hexdigest()==digest
    assert max((OUT/'sensors'/sensor/'selection.json').stat().st_mtime for sensor in freeze['selections'])<min(f.stat().st_mtime for f in files)
    daily={t:pd.read_csv(BASE/f'tomb{t}_daily.csv',index_col=0,parse_dates=True) for t in [1,2,3]}
    reproduced=0;selected=[];zero_initial=0
    for f in files:
        p=pd.read_csv(f,parse_dates=['origin_time','target_time']);r=p.iloc[0];d=daily[int(r.tomb)]
        folder=f.parents[2];selection=load_json(folder/'selection.json')
        assert selection['selected']==min(selection['trials'],key=lambda q:q['score'])
        assert len(selection['trials'])==8
        for trial in selection['trials']:
            vals=[]
            for year in [2022,2023]:
                vp=folder/'search'/trial['config']['name']/str(year)
                v=pd.read_csv(vp/'validation.csv');base=v.actual_delta.abs().groupby(v.horizon_days).mean().clip(lower=1e-10)
                score=float(((v.predicted_delta-v.actual_delta).abs().groupby(v.horizon_days).mean()/base).mean())
                assert np.isclose(score,load_json(vp/'complete.json')['score'],rtol=1e-5,atol=1e-6)
                vals.append(score)
            assert np.isclose(np.mean(vals),trial['score'],rtol=1e-5,atol=1e-6)
        old=pd.read_csv(BASE/'runs'/f't{r.tomb}_{r.sensor}_self_lstm_{r.seed}'/'predictions.csv',parse_dates=['origin_time','target_time'])
        assert np.array_equal(p[['origin_time','target_time','horizon_days']].to_numpy(),old[['origin_time','target_time','horizon_days']].to_numpy())
        assert np.allclose(p.actual,old.actual)
        assert np.allclose(p.origin,old.origin)
        assert np.allclose(p.guarded,p.origin+p.alpha*(p.optimized-p.origin))
        assert np.allclose(p.actual,d.loc[p.target_time,r.sensor])
        meta=load_json(f.parent/'metadata.json');cfg=meta['config']
        tr=np.array(meta['train_indices']);va=np.array(meta['validation_indices'])
        assert (d.index[tr+30]<pd.Timestamp('2023-01-01')).all()
        assert (d.index[va]>=pd.Timestamp('2023-01-01')).all()
        assert (d.index[va+30]<pd.Timestamp('2024-01-01')).all()
        assert folder.joinpath('selection.json').stat().st_mtime<=f.stat().st_mtime
        net=PatchTransformer(len(meta['inputs']),cfg['days'],cfg['width'])
        net.load_state_dict(torch.load(f.parent/'weights.pt',weights_only=True));net.eval()
        t=int(d.index.get_loc(p.origin_time.iloc[0]));scale=np.array(meta['feature_scale'])
        x=build_inputs(d,[t],meta['inputs'],cfg['days'],scale)
        changed=d.copy();changed.iloc[t+1:]=1e9
        xx=build_inputs(changed,[t],meta['inputs'],cfg['days'],scale);assert torch.equal(x,xx)
        with torch.no_grad():pred=float(p.origin.iloc[0])+net(x).numpy()[0]*np.array(meta['target_scale'])
        expected=p[p.origin_time==d.index[t]].sort_values('horizon_days').optimized.to_numpy()
        assert np.allclose(pred,expected,rtol=1e-5,atol=1e-8)
        info=load_json(f.parent/'complete.json');zero_initial+=int(info['best_epoch']==0);reproduced+=1
        if r.seed==42:selected.append(dict(tomb=int(r.tomb),sensor=r.sensor,**cfg,
            validation_score=selection['selected']['score'],alpha_7=selection['alpha']['7'],alpha_30=selection['alpha']['30'],
            parameters=info['parameters'],best_epoch_seed42=info['best_epoch']))
    pd.DataFrame(selected).to_csv(OUT/'selected_settings.csv',index=False)
    save_json(OUT/'audit.json',dict(runs=66,search_runs=352,reproduced=reproduced,
        same_test_origins_as_previous=True,selection_recomputed_from_2022_2023=True,
        future_input_perturbation_passed=True,initial_zero_change_models=zero_initial))

def export():
    allm=pd.read_csv(OUT/'comparison_all.csv');selected=pd.read_csv(OUT/'selected_settings.csv')
    preds=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['origin_time','target_time'])
    oldpred=pd.read_csv(BASE/'predictions_all.csv.gz',parse_dates=['origin_time','target_time'])
    tables=[];summary=[]
    for h in H:
        m=allm[allm.horizon_days==h]
        table=m.pivot(index=['tomb','sensor'],columns='model',values='MAE').reset_index()
        base=m[m.model=='optimized'][['tomb','sensor','baseline_MAE','n']]
        table=table.merge(base,on=['tomb','sensor']);table['horizon_days']=h
        table['improvement_vs_old_environment_pct']=100*(1-table.optimized/table.transformer_environment)
        table['improvement_vs_old_self_pct']=100*(1-table.optimized/table.transformer_self)
        table['improvement_vs_persistence_pct']=100*(1-table.optimized/table.baseline_MAE)
        tables.append(table)
        summary.append(dict(horizon_days=h,sensors=22,
            improved_vs_old_self=int((table.optimized<table.transformer_self-1e-12).sum()),
            improved_vs_old_environment=int((table.optimized<table.transformer_environment-1e-12).sum()),
            beats_old_lstm_environment=int((table.optimized<table.lstm_environment-1e-12).sum()),
            pure_beats_persistence=int((table.optimized<table.baseline_MAE-1e-12).sum()),
            pure_equals_persistence=int(np.isclose(table.optimized,table.baseline_MAE,rtol=1e-8,atol=1e-12).sum()),
            guarded_beats_persistence=int((table.guarded<table.baseline_MAE-1e-12).sum()),
            median_sensor_error_reduction_vs_old_env_pct=float(table.improvement_vs_old_environment_pct.median())))
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(OUT/'sensor_comparison.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'summary.json',summary)
    with pd.ExcelWriter(OUT/'Transformer_optimization_comparison.xlsx') as w:
        table.to_excel(w,sheet_name='Comparison MAE',index=False)
        allm.to_excel(w,sheet_name='MAE and RMSE',index=False)
        selected.to_excel(w,sheet_name='Selected settings',index=False)
        pd.read_csv(OUT/'metrics_by_seed.csv').to_excel(w,sheet_name='Independent seeds',index=False)
        pd.DataFrame([
            ('최적화 대상','Transformer만 최적화; 기존 LSTM은 고정 설정 비교 기준이므로 구조 자체의 우열 실험이 아님'),
            ('검색','센서별 8개 후보 × 2022/2023 검증; 과거 30/90/180일, 센서 단독/내부 환경 입력, 폭 16/32'),
            ('변경','현재값 기준 입력 중심화, 5일 패치, 1층 소형 Transformer, Huber 손실, AdamW, 초기 변화량 0'),
            ('결측','과거 30일의 기존 평가 가능 날짜 유지. 더 오래된 결측값은 0 자리표시자와 관측 마스크로 표시; 보간하지 않음'),
            ('선택','2022년과 2023년의 7/30일 상대 MAE를 동일 가중 평균; 2024년 결과로 후보를 선택하지 않음'),
            ('학습','2023년 이전 목표값으로 학습, 2023년 검증으로 조기 종료. 선택한 설정을 초기값 42/43/44로 반복'),
            ('시험','기존 실험과 같은 2024년 예측 시작일·관측 정답. 이 기간은 이미 이전 분석에서 확인했으므로 독립적인 새 시험자료가 아님'),
            ('optimized','최적화한 신경망 자체의 결과'),
            ('guarded','검증으로 선택한 0~1 계수를 신경망 변화량에 곱한 별도 운영 후보. 계수가 0이면 현재값 유지이며 Transformer 개선으로 세지 않음'),
            ('수치','MAE/RMSE는 독립 3회 평균. 센서 간 물리 단위의 오차를 합산하지 않음. 그래프는 seed 42 예시'),
        ],columns=['항목','설명']).to_excel(w,sheet_name='Method',index=False)
        for sheet in w.book.worksheets:
            sheet.freeze_panes='C2';sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:sheet.column_dimensions[col[0].column_letter].width=min(45,max(15,len(str(col[0].value))+2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    sensors=selected.sort_values(['tomb','sensor']).sensor.tolist()
    fig,axs=plt.subplots(1,2,figsize=(15,9),layout='constrained')
    for ax,h in zip(axs,H):
        g=table[table.horizon_days==h].set_index('sensor').reindex(sensors);y=np.arange(len(g))
        for offset,col,color,label in [(-.25,'transformer_environment','#a6a6a6','Original Transformer + environment'),(0,'optimized','#176bba','Optimized Transformer'),(.25,'lstm_environment','#d96a20','Original LSTM + environment')]:
            ax.scatter(g[col]/g.baseline_MAE,y+offset,s=20,color=color,label=label)
        ax.axvline(1,color='black',ls='--',lw=1);ax.set_xscale('log');ax.set_yticks(y,sensors);ax.invert_yaxis()
        ax.set_title(f'{h}-day forecast');ax.set_xlabel('MAE / persistence MAE (below 1 is better)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Transformer optimization | mean MAE across 3 seeds | 2024 retrospective audit')
    fig.savefig(OUT/'optimization_overview.png',dpi=180);fig.savefig(OUT/'optimization_overview.pdf');plt.close(fig)
    for h in H:
        fig,axs=plt.subplots(11,2,figsize=(15,28),layout='constrained')
        for ax,sensor in zip(axs.flat,sensors):
            a=preds[(preds.sensor==sensor)&(preds.seed==42)&(preds.horizon_days==h)].set_index('target_time').sort_index()
            b=oldpred[(oldpred.sensor==sensor)&(oldpred.seed==42)&(oldpred.horizon_days==h)&(oldpred['mode']=='environment')&(oldpred.model=='transformer')].set_index('target_time').sort_index()
            dates=pd.date_range(a.index.min(),a.index.max(),freq='D');a=a.reindex(dates);b=b.reindex(dates)
            ax.plot(dates,a.actual-a.origin,color='#222222',lw=1.1,label='Observed change')
            ax.plot(dates,b.prediction-b.origin,color='#999999',lw=.8,label='Original Transformer + environment')
            ax.plot(dates,a.optimized-a.origin,color='#176bba',lw=1,label='Optimized Transformer')
            ax.axhline(0,color='#d96a20',ls='--',lw=.6,label='Persistence (zero change)')
            ax.set_title(sensor,loc='left',weight='bold');ax.set_ylabel('change (mm)' if '-D' in sensor else 'change (degree)')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(alpha=.15)
        handles,labels=axs.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.978),ncol=4)
        fig.suptitle(f'{h}-day change | before / after Transformer optimization | seed 42\nEach point is a separate forecast issued {h} days before the target date',fontsize=13,y=.998)
        fig.get_layout_engine().set(rect=(0,0,1,.963))
        fig.savefig(OUT/f'all_sensors_{h}d_change.png',dpi=180);fig.savefig(OUT/f'all_sensors_{h}d_change.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':audit();export()
