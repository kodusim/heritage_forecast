from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib as mpl
from .storage import write_json,sha,now

@dataclass
class FigureSpec:
    kind:str='series'
    width_mm:float=180
    height_mm:float=100
    dpi:int=300
    font_size:float=9
    font_family:str='Malgun Gothic'
    title:str=''
    xlabel:str='Time'
    ylabel:str='Value'
    grayscale:bool=False
    legend:bool=True

def make_figure(data,spec:FigureSpec):
    if not 60<=spec.width_mm<=400 or not 40<=spec.height_mm<=400:raise ValueError('그림 크기 범위를 확인하세요.')
    if data.empty:raise ValueError('그림에 사용할 자료가 없습니다.')
    if spec.kind in ('series','correlation') and not np.isfinite(data.select_dtypes(include='number').to_numpy()).any():raise ValueError('유효한 수치 관측값이 없습니다.')
    with mpl.rc_context({'font.family':[spec.font_family,'DejaVu Sans'],'font.size':spec.font_size,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        fig=Figure(figsize=(spec.width_mm/25.4,spec.height_mm/25.4),dpi=110,layout='constrained')
        ax=fig.add_subplot(111)
        colors=['#222222','#777777','#aaaaaa','#444444'] if spec.grayscale else ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
        if spec.kind=='series':
            for i,c in enumerate(data.select_dtypes(include='number')):
                ax.plot(data.index,data[c],label=c,color=colors[i%len(colors)],lw=.9,linestyle=['-','--','-.',':'][i%4])
        elif spec.kind=='comparison':
            if not {'time','actual','predicted','model'}<=set(data):raise ValueError('비교 그래프에는 학습 결과가 필요합니다.')
            keys=['model','mode'] if 'mode' in data else ['model']
            actual=data.groupby('time').actual.first();actual.index=pd.to_datetime(actual.index)
            timeline=pd.date_range(actual.index.min(),actual.index.max(),freq='h');actual=actual.reindex(timeline)
            ax.plot(actual.index,actual,color='#333333',lw=1.1,label='Observed')
            for i,(key,g) in enumerate(data.groupby(keys)):
                g=g.copy();g.index=pd.to_datetime(g.time);line=g.predicted.reindex(timeline)
                ax.plot(line.index,line,label=' / '.join(key) if isinstance(key,tuple) else key,color=colors[i%len(colors)],lw=.85,linestyle=['--','-.',':','-'][i%4])
        elif spec.kind=='scatter':
            clean=data[['actual','predicted']].dropna()
            if clean.empty:raise ValueError('관측된 실제값이 없습니다.')
            ax.scatter(clean.actual,clean.predicted,s=8,alpha=.35,color=colors[0],rasterized=True)
            lo=clean.min().min();hi=clean.max().max();ax.plot([lo,hi],[lo,hi],color='#555555',ls='--',lw=.8,label='1:1')
        elif spec.kind=='residual':
            if not {'actual','predicted','time'}<=set(data):raise ValueError('잔차에는 실제값이 있는 학습 결과가 필요합니다.')
            residual=pd.Series((data.predicted-data.actual).to_numpy(),index=pd.to_datetime(data.time)).sort_index()
            residual=residual.reindex(pd.date_range(residual.index.min(),residual.index.max(),freq='h'))
            ax.plot(residual.index,residual,lw=.7,color=colors[0],label='Predicted - observed');ax.axhline(0,color='#777777',ls='--',lw=.7)
        elif spec.kind=='horizon':
            for i,(key,g) in enumerate(data.groupby(['model','mode'])):
                g=g.sort_values('horizon');ax.plot(g.horizon,g.RMSE,marker='o',ms=3,lw=1,color=colors[i%len(colors)],label=' / '.join(key))
        elif spec.kind in ('correlation','missing'):
            if spec.kind=='correlation':
                matrix=data.select_dtypes(include='number').corr();im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm' if not spec.grayscale else 'gray');labs=matrix.columns
                ax.set_xticks(range(len(labs)),labs,rotation=45,ha='right');ax.set_yticks(range(len(labs)),labs);fig.colorbar(im,ax=ax,label='Pearson r',shrink=.8)
            else:
                matrix=data.isna().resample('MS').mean().T*100;im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='YlOrRd' if not spec.grayscale else 'gray_r')
                stride=max(1,len(matrix.columns)//12);ax.set_xticks(range(0,len(matrix.columns),stride),[d.strftime('%Y-%m') for d in matrix.columns[::stride]],rotation=45,ha='right');ax.set_yticks(range(len(matrix)),matrix.index);fig.colorbar(im,ax=ax,label='Missing (%)',shrink=.8)
        else:raise ValueError('지원하지 않는 Figure 종류입니다.')
        ax.set_title(spec.title,pad=10,fontsize=spec.font_size+1)
        if spec.kind not in ('correlation','missing'):
            ax.set_xlabel(spec.xlabel);ax.set_ylabel(spec.ylabel);ax.grid(alpha=.18,linewidth=.5)
            ax.spines[['top','right']].set_visible(False)
            if spec.legend and ax.get_legend_handles_labels()[0]:ax.legend(frameon=False,fontsize=max(6,spec.font_size-1),loc='best')
        return fig

def export_figure(data,spec,destination,provenance):
    path=Path(destination)
    if path.suffix.lower() not in ['.png','.pdf','.svg','.tiff']:raise ValueError('PNG / PDF / SVG / TIFF를 선택하세요.')
    if spec.dpi<=0 or spec.width_mm/25.4*spec.height_mm/25.4*spec.dpi**2>80_000_000:raise ValueError('그림이 너무 큽니다. 크기 또는 해상도를 낮춰 주세요 (최대 8천만 화소).')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig=make_figure(data,spec)
    with mpl.rc_context({'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False}):
        fig.savefig(path,dpi=spec.dpi,facecolor='white')
    data_path=path.with_name(path.stem+'_data.csv.gz');data.to_csv(data_path,compression='gzip',encoding='utf-8-sig')
    meta=dict(provenance,spec=asdict(spec),generated_at=now(),data_file=data_path.name,data_sha256=sha(data_path),figure_sha256=sha(path),
              notes='No smoothing or interpolation. Missing observations remain missing. Correlation is exploratory, not causation.')
    write_json(path.with_name(path.stem+'_figure.json'),meta)
    return path
