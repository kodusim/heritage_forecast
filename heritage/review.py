"""Results-first, multi-sensor review. All predictions come from held-out runs."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.dates import AutoDateLocator,ConciseDateFormatter
from .storage import projects,write_json,sha,now
from .data import hourly
from .engine import metrics

COLORS={'actual':'#243247','environment':'#0079bf','self':'#dc792b'}
MODE_NAMES={'environment':'거동 + 내부 온도','self':'거동 과거값만'}

@dataclass
class Review:
    project:object
    group:str
    version:dict
    run:dict|None
    observed:pd.DataFrame
    predictions:pd.DataFrame

@dataclass
class Panel:
    review:Review
    sensor:str
    unit:str
    observed:pd.Series
    lines:dict
    scores:dict
    reason:str
    start:pd.Timestamp
    end:pd.Timestamp

def review_projects():
    candidates=projects()
    preferred=[p for p in candidates if p.setting('customer_review')]
    others=[p for p in candidates if p not in preferred and any(r.get('status')=='complete' and 'transformer' in r.get('config',{}).get('models',[]) for r in p.list('run'))]
    return sorted(preferred,key=lambda p:int(p.setting('review_tomb','0')))+others

def load_review(project):
    runs=[r for r in project.list('run') if r.get('status')=='complete' and 'transformer' in r.get('config',{}).get('models',[])]
    run=runs[0] if runs else None
    raw,version=project.load_data(run['config']['dataset_id'] if run else None)
    observed=hourly(raw,minimum=run['config'].get('minimum',4) if run else 4,metadata=version['columns'])
    predictions=pd.DataFrame()
    if run:
        chunks=[]
        for chunk in pd.read_csv(project.file(run['folder']+'/predictions.csv.gz'),chunksize=200000,parse_dates=['time','origin']):
            selected=chunk[(chunk['split']=='test')&(chunk['model']=='transformer')]
            if len(selected):chunks.append(selected)
        if chunks:predictions=pd.concat(chunks,ignore_index=True)
    tomb=project.setting('review_tomb')
    return Review(project,f'{tomb}번 고분' if tomb else project.name,version,run,observed,predictions)

def make_panels(reviews,horizon=24,mode='environment',days=0):
    panels=[];modes=['environment','self'] if mode=='both' else [mode]
    for review in reviews:
        frame=review.observed
        start=pd.Timestamp(review.run['config']['val_end'])+pd.Timedelta(hours=1) if review.run else frame.index.min()
        end=frame.index.max()
        if days:start=max(start,end-pd.Timedelta(days=days))
        for sensor,meta in review.version['columns'].items():
            if meta['role']!='target':continue
            observed=frame[sensor].loc[start:end];lines={};scores={};reason=''
            table=review.predictions
            for condition in modes:
                if table.empty:continue
                selected=table[(table.sensor==sensor)&(table.horizon==horizon)&(table['mode']==condition)&(table.time>=start)&(table.time<=end)].sort_values('time')
                if selected.empty:continue
                if selected.time.duplicated().any():raise ValueError(f'{sensor}: 같은 평가 시각에 중복 예측이 있습니다.')
                line=selected.set_index('time').predicted.reindex(observed.index)
                scores[condition]=metrics(selected.actual,selected.predicted)
                baseline=metrics(selected.actual,selected.baseline)
                scores[condition]['baseline_MAE']=baseline['MAE']
                scores[condition]['improvement_pct']=100*(1-scores[condition]['MAE']/baseline['MAE']) if baseline['MAE'] else None
                lines[condition]=line
            if not lines:
                if not frame[sensor].notna().any():reason='2024년 유효 관측값 없음\n실제값·예측값을 만들지 않았습니다.'
                elif not review.run:reason='Transformer 학습 결과 준비 중'
                elif any(s['sensor']==sensor for s in review.run.get('skipped',[])):reason='학습·검증·테스트의 유효 구간 부족'
                else:reason='이 조건·기간의 저장된 예측 결과 없음'
            panels.append(Panel(review,sensor,meta['unit'],observed,lines,scores,reason,start,end))
    return panels

def metric_label(score,unit):
    if not score:return '예측 결과 없음'
    improvement=score.get('improvement_pct')
    comparison='기준선 비교 불가' if improvement is None else f'기준선보다 MAE {abs(improvement):.1f}% '+('낮음' if improvement>=0 else '높음')
    return f'MAE {score["MAE"]:.3g} {unit} · RMSE {score["RMSE"]:.3g} {unit} · {comparison}'

def draw_panel(ax,panel,compact=False):
    ax.set_facecolor('white')
    if panel.observed.notna().any():
        ax.plot(panel.observed.index,panel.observed,color=COLORS['actual'],lw=.9 if compact else 1.05,label='실제값',alpha=.9)
    for condition,line in panel.lines.items():
        ax.plot(line.index,line,color=COLORS[condition],lw=.8 if compact else 1.0,ls='-' if condition=='environment' else '--',label='Transformer · '+MODE_NAMES[condition],alpha=.9)
    ax.set_ylabel(panel.unit,labelpad=4)
    locator=AutoDateLocator(minticks=3,maxticks=4 if compact else 5);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.grid(color='#e4eaf0',linewidth=.5);ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#cbd5e1');ax.tick_params(colors='#58677a',labelsize=7 if compact else 8,length=2)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
    ax.yaxis.get_offset_text().set_fontsize(7)
    if panel.end>panel.start:ax.set_xlim(panel.start,panel.end)
    if panel.reason:
        if not panel.observed.notna().any():ax.set_yticks([])
        ax.text(.5,.52,panel.reason,transform=ax.transAxes,ha='center',va='center',color='#718096',fontsize=8 if compact else 10,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':8})
    return ax

def panel_figure(panel):
    with mpl.rc_context({'font.family':['Malgun Gothic','DejaVu Sans'],'font.size':9,'axes.unicode_minus':False}):
        fig=Figure(figsize=(5,2),dpi=100,layout='constrained');draw_panel(fig.add_subplot(111),panel)
    return fig

def overview_figure(panels,horizon=24,screen=False):
    if not panels:raise ValueError('표시할 센서가 없습니다.')
    cols=3 if len(panels)>8 else 2;rows=math.ceil(len(panels)/cols)
    with mpl.rc_context({'font.family':['Malgun Gothic','DejaVu Sans'],'font.size':8,'axes.unicode_minus':False}):
        fig=Figure(figsize=(13.6,rows*2.5+1),dpi=130,layout='constrained')
        axes=fig.subplots(rows,cols,squeeze=False)
        for ax,panel in zip(axes.flat,panels):
            draw_panel(ax,panel,compact=True)
            score=panel.scores.get('environment') or panel.scores.get('self')
            improvement=score.get('improvement_pct') if score else None
            short=f'MAE {score["MAE"]:.2g} · RMSE {score["RMSE"]:.2g}' if score else '유효 예측 결과 없음'
            if improvement is not None:short+=f' · MAE 개선 {improvement:+.1f}%'
            if len(panel.lines)>1:short='[온도 포함] '+short
            ax.set_title(f'{panel.review.group} · {panel.sensor} [{panel.unit}]\n'+short,loc='left',fontsize=8 if screen else 9,pad=5)
            ax._review_panel=panel
        for ax in list(axes.flat)[len(panels):]:ax.set_visible(False)
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=COLORS['actual'],lw=1.5,label='실제값')]
        for condition in ('environment','self'):
            if any(condition in p.lines for p in panels):handles.append(Line2D([0],[0],color=COLORS[condition],lw=1.3,ls='-' if condition=='environment' else '--',label='Transformer · '+MODE_NAMES[condition]))
        if not screen:
            fig.suptitle(f'Transformer · 실제값과 {horizon}시간 앞 예측 비교\n학습에 사용하지 않은 테스트 기간 · 매 시각까지의 관측으로 반복 예측',fontsize=13)
            fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),frameon=False,fontsize=10)
    return fig

def export_overview(panels,horizon,destination):
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower() not in ('.png','.pdf','.svg'):raise ValueError('PNG, PDF 또는 SVG로 저장하세요.')
    fig=overview_figure(panels,horizon)
    with mpl.rc_context({'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False}):fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
    records=[];summaries=[];runs={}
    for panel in panels:
        records.append(pd.DataFrame({'project':panel.review.group,'project_id':panel.review.project.path.name,'sensor':panel.sensor,'unit':panel.unit,'time':panel.observed.index,'actual':panel.observed.to_numpy(),
                      **{f'transformer_{k}':v.to_numpy() for k,v in panel.lines.items()}}))
        for mode,score in panel.scores.items():summaries.append({'project':panel.review.group,'sensor':panel.sensor,'unit':panel.unit,'mode':mode,'horizon':horizon,**score})
        if panel.review.run:runs[panel.review.run['id']]={'project_id':panel.review.project.path.name,'dataset_id':panel.review.version['id'],'config':panel.review.run['config']}
    data_path=path.with_name(path.stem+'_data.csv.gz');pd.concat(records,ignore_index=True).to_csv(data_path,index=False,compression='gzip')
    pd.DataFrame(summaries).to_csv(path.with_name(path.stem+'_metrics.csv'),index=False,encoding='utf-8-sig')
    write_json(path.with_name(path.stem+'_figure.json'),{'generated_at':now(),'horizon':horizon,'runs':runs,'sensors':[{'project':p.review.group,'sensor':p.sensor,'reason':p.reason,'start':str(p.start),'end':str(p.end)} for p in panels],
               'figure_sha256':sha(path),'data_sha256':sha(data_path),'notes':'Held-out rolling-origin evaluation; observed past inputs are updated at each origin. Missing data are not interpolated. Baseline is last observed value.'})
    return path
