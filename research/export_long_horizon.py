"""Publication export of matched long-horizon experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .compare_long_horizon import OUT

def export():
    m=pd.read_csv(OUT/'comparison.csv')
    seeds=pd.read_csv(OUT/'metrics_by_seed.csv')
    preds=pd.read_csv(OUT/'predictions_all.csv.gz',parse_dates=['origin_time','target_time'])
    counts=pd.concat([pd.read_csv(OUT/f'sample_counts_tomb{i}.csv') for i in [1,2,3]],ignore_index=True)
    counts.to_csv(OUT/'sample_counts.csv',index=False)
    rows=[]
    for (tomb,sensor,mode,h),g in m.groupby(['tomb','sensor','mode','horizon_days']):
        a=g.set_index('model');l=a.loc['lstm'];t=a.loc['transformer']
        rows.append(dict(tomb=tomb,sensor=sensor,mode=mode,horizon_days=h,n=int(l.n),
            lstm_MAE=l.MAE,transformer_MAE=t.MAE,persistence_MAE=l.baseline_MAE,
            lstm_RMSE=l.RMSE,transformer_RMSE=t.RMSE,
            transformer_improvement_vs_lstm_pct=100*(1-t.MAE/l.MAE),
            winner='LSTM' if l.MAE<t.MAE else 'Transformer',
            lstm_beats_persistence=l.MAE<l.baseline_MAE,
            transformer_beats_persistence=t.MAE<t.baseline_MAE))
    table=pd.DataFrame(rows);table.to_csv(OUT/'sensor_comparison.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(OUT/'LSTM_Transformer_7d_30d.xlsx') as w:
        table.to_excel(w,sheet_name='Comparison',index=False)
        seeds.to_excel(w,sheet_name='All seeds',index=False)
        counts.to_excel(w,sheet_name='Samples',index=False)
        pd.DataFrame([
            ('대상','1·2·3번 고분, 센서 22개; 6-D3는 검증·시험 관측 부족으로 제외'),
            ('원자료','기존 감사에서 연도별 시트를 합치고 중복 충돌값을 제외한 캐시; 제공된 거동 데이터의 사전 노이즈 제거 방식은 별도 검증하지 않음'),
            ('일 집계','10분 관측 144개 중 최소 96개가 있는 날의 평균; 하루가 끝난 시각으로 표기; 보간 없음'),
            ('예측','과거 30일에서 7일·30일 뒤 일평균의 현재 대비 변화량을 직접 예측'),
            ('학습','예측 대상 시각이 2023-01-01 이전인 표본'),
            ('검증','예측 시작 시각이 2023-01-01 이후이고 대상 시각이 2024-01-01 이전인 표본'),
            ('시험','2024년 예측 시작 시각; 7일·30일 모두 실제 관측 정답이 있는 동일 표본'),
            ('환경','과거 내부 주 센서·예비 센서 온도 두 열; 미래 기상 실측·예보는 입력하지 않음; 장기간 센서 위치·표기 변화가 존재'),
            ('모델','기존 heritage_forecast의 LSTM/Transformer 클래스 재사용; 센서별 2개 예측기간 출력으로 새로 학습; 기존 가중치 재사용 아님'),
            ('학습 설정','각 모델 64차원·2층, dropout 0.2, Adam 0.001, 최대 40 epoch, 검증 손실 기준 patience 7; 초기값 42·43·44'),
            ('표 수치','3회 독립 학습의 MAE/RMSE 평균; 단위가 다른 센서의 물리 오차는 합산하지 않음'),
            ('그래프','사전 지정 seed 42 예시; 각 날짜마다 과거에 발행한 별도 예측들을 연결함; 하나의 시작점에서 생성한 미래 전체 경로가 아님'),
            ('한계','2024년 단일 시험 기간의 고정 설정 비교; 겹치는 예측 구간은 통계적으로 독립 아님; 모델 전체의 보편적 우월성이나 손상 여부를 입증하지 않음'),
        ],columns=['항목','설명']).to_excel(w,sheet_name='Method',index=False)
        for sheet in w.book.worksheets:
            sheet.freeze_panes='C2';sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns: sheet.column_dimensions[col[0].column_letter].width=min(40,max(14,len(str(col[0].value))+2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    sensors=counts[counts.included].sensor.tolist()
    # Representative, pre-specified seed 42; tables average independent seeds, not an ensemble.
    for h in [7,30]:
        for mode in ['self','environment']:
            fig,axs=plt.subplots(11,2,figsize=(15,28),layout='constrained')
            for ax,sensor in zip(axs.flat,sensors):
                g=preds[(preds.sensor==sensor)&(preds.horizon_days==h)&(preds['mode']==mode)&(preds.seed==42)]
                a=g[g.model=='lstm'].set_index('target_time').sort_index()
                b=g[g.model=='transformer'].set_index('target_time').sort_index()
                dates=pd.date_range(a.index.min(),a.index.max(),freq='D')
                a=a.reindex(dates);b=b.reindex(dates)
                ax.plot(dates,a.actual,color='#222222',lw=1.2,label='Observed')
                ax.plot(dates,a.prediction,color='#d96a20',lw=1,label='LSTM')
                ax.plot(dates,b.prediction,color='#176bba',lw=1,label='Transformer')
                ax.plot(dates,a.persistence,color='#888888',lw=.7,ls='--',label='Persistence')
                ax.set_title(sensor,loc='left',weight='bold');ax.set_ylabel('mm' if '-D' in sensor else 'degree')
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.grid(alpha=.15)
            handles,labels=axs.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.978),ncol=4)
            fig.suptitle(f'{h}-day-ahead daily mean | {mode} inputs | held-out 2024 | seed 42\nForecasts issued at successive daily origins; gaps are not interpolated',fontsize=13,y=.998)
            fig.get_layout_engine().set(rect=(0,0,1,.963))
            name=f'all_sensors_{h}d_{mode}'
            fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
    # Dimensionless performance comparison, keeping physical units separate.
    fig,axs=plt.subplots(2,2,figsize=(15,14),layout='constrained')
    for i,mode in enumerate(['self','environment']):
        for j,h in enumerate([7,30]):
            ax=axs[i,j];g=table[(table['mode']==mode)&(table.horizon_days==h)].set_index('sensor').reindex(sensors)
            y=np.arange(len(g))
            ax.barh(y-.18,g.lstm_MAE/g.persistence_MAE,.34,color='#d96a20',label='LSTM')
            ax.barh(y+.18,g.transformer_MAE/g.persistence_MAE,.34,color='#176bba',label='Transformer')
            ax.axvline(1,color='black',ls='--',lw=1)
            ax.set_yticks(y,sensors);ax.invert_yaxis();ax.set_xscale('log')
            ax.set_title(f'{mode} | {h} days');ax.set_xlabel('MAE / persistence MAE (below 1 is better)');ax.legend()
    fig.suptitle('Mean error across 3 independent training seeds | 2024 holdout',fontsize=14)
    fig.savefig(OUT/'comparison_overview.png',dpi=180);fig.savefig(OUT/'comparison_overview.pdf');plt.close(fig)
    fig,axs=plt.subplots(11,2,figsize=(15,28),layout='constrained')
    for ax,sensor in zip(axs.flat,sensors):
        g=preds[(preds.sensor==sensor)&(preds.horizon_days==30)&(preds['mode']=='environment')&(preds.seed==42)]
        a=g[g.model=='lstm'].set_index('target_time').sort_index()
        b=g[g.model=='transformer'].set_index('target_time').sort_index()
        dates=pd.date_range(a.index.min(),a.index.max(),freq='D');a=a.reindex(dates);b=b.reindex(dates)
        ax.plot(dates,a.actual-a.origin,color='#222222',lw=1.2,label='Observed 30-day change')
        ax.plot(dates,a.prediction-a.origin,color='#d96a20',lw=1,label='LSTM')
        ax.plot(dates,b.prediction-b.origin,color='#176bba',lw=1,label='Transformer')
        ax.axhline(0,color='#888888',lw=.7,ls='--',label='Persistence (zero change)')
        ax.set_title(sensor,loc='left',weight='bold');ax.set_ylabel('change (mm)' if '-D' in sensor else 'change (degree)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(alpha=.15)
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.978),ncol=4)
    fig.suptitle('30-day change from forecast origin | environment inputs | seed 42\nEach point is a separate forecast issued 30 days before its target date',fontsize=13,y=.998)
    fig.get_layout_engine().set(rect=(0,0,1,.963))
    fig.savefig(OUT/'all_sensors_30d_change.png',dpi=180);fig.savefig(OUT/'all_sensors_30d_change.pdf');plt.close(fig)
    summary=[]
    for (mode,h),g in table.groupby(['mode','horizon_days']):
        paired=seeds[(seeds['mode']==mode)&(seeds.horizon_days==h)].pivot(index=['sensor','seed'],columns='model',values='MAE')
        all_lstm=(paired.lstm<paired.transformer).groupby(level='sensor').all().sum()
        all_tf=(paired.transformer<paired.lstm).groupby(level='sensor').all().sum()
        summary.append(dict(mode=mode,horizon_days=int(h),sensors=len(g),LSTM_wins=int((g.winner=='LSTM').sum()),
            Transformer_wins=int((g.winner=='Transformer').sum()),
            LSTM_wins_all_seeds=int(all_lstm),Transformer_wins_all_seeds=int(all_tf),
            LSTM_beats_persistence=int(g.lstm_beats_persistence.sum()),Transformer_beats_persistence=int(g.transformer_beats_persistence.sum())))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':export()
