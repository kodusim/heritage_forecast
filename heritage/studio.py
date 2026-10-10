"""Task-oriented desktop interface; archived datasets and numerical engines stay intact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QDateTime, QDate
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QFormLayout,
    QComboBox,QCheckBox,QListWidget,QSpinBox,QLineEdit,QTableView,QDateTimeEdit,
    QDateEdit,QTabWidget,QSplitter,QDialog,QDialogButtonBox,QMessageBox,QGroupBox)
from .ui import MainWindow as LegacyWindow, label,button,card,combo,set_table
from .storage import read_json
from .history import registrations,disabled,affected

MODEL={'lstm':'LSTM','transformer':'Transformer','linear':'선형 모델','persistence':'직전 값 유지'}
MODE={'self':'센서 이력만','environment':'환경 포함'}
STATUS={'complete':'완료','running':'진행 중','failed':'실패','cancelled':'취소','interrupted':'중단'}


class DateField(QDateTimeEdit):
    def __init__(self):
        super().__init__();self.setCalendarPopup(True);self.setDisplayFormat('yyyy-MM-dd HH:mm');self.setDateTime(QDateTime.currentDateTime())
    def setText(self,value):
        self.setDateTime(QDateTime.fromString(str(value)[:19],'yyyy-MM-dd HH:mm:ss'))
    def text(self):return self.dateTime().toString('yyyy-MM-dd HH:mm:ss')


class OptionalDate(QWidget):
    def __init__(self):
        super().__init__();v=QHBoxLayout(self);v.setContentsMargins(0,0,0,0)
        self.check=QCheckBox('기간 지정');self.date=DateField();self.date.setEnabled(False)
        self.check.toggled.connect(self.date.setEnabled);v.addWidget(self.check);v.addWidget(self.date)
    def text(self):return self.date.text() if self.check.isChecked() else ''


def fold(title):
    host=QWidget();v=QVBoxLayout(host);v.setContentsMargins(0,0,0,0)
    toggle=QCheckBox(title);body=QWidget();body.setVisible(False);toggle.toggled.connect(body.setVisible)
    v.addWidget(toggle);v.addWidget(body);return host,QVBoxLayout(body)


class MainWindow(LegacyWindow):
    def __init__(self):
        self.valid_times=pd.DatetimeIndex([]);self.valid_model=None;self.origins_job=None
        super().__init__()
        self.setWindowTitle('Heritage Studio · 문화유산 예측');self.resize(1440,920)
        icon=Path(__file__).parent/'assets'/'heritage.png';self.setWindowIcon(QIcon(str(icon)))
        sidebar=self.nav[0].parentWidget().layout()
        mark=label('');mark.setPixmap(QPixmap(str(icon)).scaled(76,76,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));sidebar.insertWidget(0,mark)
        self.build_results()
        b=button('03   결과 비교',lambda:self.navigate(5));b.setObjectName('nav');b.setCheckable(True);self.nav.append(b)
        sidebar.insertWidget(sidebar.indexOf(self.nav[3]),b)
        for i,t in enumerate(['시작하기','01   데이터','02   학습하기','04   새 예측','05   논문 그림']):self.nav[i].setText(t)
        self.log.hide();self.progress.hide();self.job_label.setText('준비되었습니다. 왼쪽에서 작업을 선택하세요.')
        self.log_toggle=QCheckBox('작업 상세');self.log_toggle.toggled.connect(self.log.setVisible)
        self.job_label.parentWidget().layout().itemAt(0).layout().insertWidget(1,self.log_toggle)
        self.job_label.parentWidget().setMaximumHeight(130)
        self.refresh();self.navigate(0)

    def page(self,title,description):
        if title in ('데이터 관리','새 예측','결과 비교'):
            w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(0,0,0,0);v.setSpacing(12)
            v.addWidget(label(title,'pageTitle'));v.addWidget(label(description,'muted'));self.stack.addWidget(w);return v
        return super().page(title,description)

    def build_data(self):
        v=self.page('데이터 관리','프로젝트 하나에 고분 하나의 자료를 모읍니다. 파일 하나에 여러 센서가 있어도 한 번에 등록할 수 있습니다.')
        row=QHBoxLayout();row.addWidget(button('엑셀 / CSV 추가',self.import_file,True));row.addWidget(button('품질표 저장',self.export_quality));row.addStretch();v.addLayout(row)
        self.data_summary=label('엑셀을 추가하면 센서와 관측 기간을 확인할 수 있습니다.','muted');v.addWidget(self.data_summary)
        tabs=QTabWidget();v.addWidget(tabs,1)
        q=QWidget();qv=QVBoxLayout(q);self.dataset_combo=QComboBox();qv.addWidget(label('확인할 데이터 버전'));qv.addWidget(self.dataset_combo)
        self.quality_table=QTableView();qv.addWidget(self.quality_table,1);tabs.addTab(q,'센서 · 데이터 품질')
        h=QWidget();hv=QVBoxLayout(h);hv.addWidget(label('잘못 올린 파일·시트를 선택해 등록 취소하세요. 원본 엑셀은 지우지 않으며, 취소한 등록은 다시 복원할 수 있습니다.','muted'))
        self.import_table=QTableView();hv.addWidget(self.import_table,1);rr=QHBoxLayout()
        rr.addWidget(button('선택 등록 취소',lambda:self.change_import(False)));rr.addWidget(button('선택 등록 복원',lambda:self.change_import(True)));rr.addStretch();hv.addLayout(rr);tabs.addTab(h,'등록 내역 · 삭제 / 복원')
        v.addWidget(label('거동·환경 파일은 같은 고분에 차례로 추가하세요. 다른 고분은 새 프로젝트로 구분하세요.','muted'))
        self.dataset_combo.currentIndexChanged.connect(self.show_quality)

    def build_training(self):
        v=self.page('학습하기','센서와 예측 기간을 고르고 시작하세요. 완료되면 결과 비교 화면에서 실제값과 예측값을 확인합니다.')
        c,cv=card('1. 어떤 센서를 예측할까요?');self.targets=QListWidget();self.targets.setMinimumHeight(160);self.targets.setMaximumHeight(210);cv.addWidget(self.targets)
        row=QHBoxLayout();row.addWidget(button('전체 선택',lambda:self.select_sensors(True)));row.addWidget(button('선택 해제',lambda:self.select_sensors(False)));row.addStretch();cv.addLayout(row);v.addWidget(c)
        c,cv=card('2. 얼마 뒤를 예측할까요?');self.period=combo([('1시간 · 6시간 · 하루 뒤','1,6,24'),('하루 · 3일 · 7일 뒤','24,72,168'),('7일 · 14일 · 30일 뒤','168,336,720'),('직접 설정',None)])
        self.horizons=QLineEdit('1,6,24');self.horizons.setPlaceholderText('시간 단위, 예: 24, 168');self.horizons.hide();cv.addWidget(self.period);cv.addWidget(self.horizons)
        self.period.currentIndexChanged.connect(self.period_changed)
        cv.addWidget(label('기간이 길수록 충분한 학습·검증·평가 자료가 필요합니다. 시간 단위로 집계한 데이터를 사용합니다.','muted'));v.addWidget(c)
        c,cv=card('3. 어떤 모델을 비교할까요?');row=QHBoxLayout();self.model_checks={}
        for key in ('lstm','transformer','linear'):
            q=QCheckBox(MODEL[key]);q.setChecked(key!='linear');row.addWidget(q);self.model_checks[key]=q
        row.addStretch();cv.addLayout(row);cv.addWidget(label('직전 관측값을 그대로 쓰는 단순 예측도 함께 비교합니다.','muted'))
        self.self_mode=QCheckBox('센서의 과거값으로 학습');self.self_mode.setChecked(True);self.env_mode=QCheckBox('온도·습도 등 환경을 포함한 조건도 비교');cv.addWidget(self.self_mode);cv.addWidget(self.env_mode)
        self.envs=QListWidget();self.envs.setFixedHeight(115);self.envs.hide();self.env_mode.toggled.connect(self.envs.setVisible);cv.addWidget(self.envs);v.addWidget(c)
        c,cv=card('자료를 나누는 기간');self.train_end=DateField();self.val_end=DateField();form=QFormLayout();form.addRow('학습에 사용할 마지막 날짜',self.train_end);form.addRow('모델 선택용 검증의 마지막 날짜',self.val_end);cv.addLayout(form)
        self.split_hint=label('자료의 앞 70%는 학습, 다음 15%는 검증, 나머지는 최종 비교에 쓰도록 제안합니다. 날짜는 달력에서 변경할 수 있습니다.','muted');cv.addWidget(self.split_hint);v.addWidget(c)
        advanced,av=fold('고급 설정 열기');form=QFormLayout();self.lookback=QSpinBox();self.lookback.setRange(2,720);self.lookback.setValue(24)
        self.epochs=QSpinBox();self.epochs.setRange(1,500);self.epochs.setValue(20);self.minimum=QSpinBox();self.minimum.setRange(1,60);self.minimum.setValue(4)
        self.seed=QSpinBox();self.seed.setRange(0,100000);self.seed.setValue(42);self.device=combo([('CPU','cpu'),('NVIDIA CUDA','cuda')])
        for title,w in [('과거 몇 시간의 값을 참고할까요?',self.lookback),('최대 학습 반복 횟수',self.epochs),('시간당 필요한 최소 관측 수',self.minimum),('재현용 난수 값',self.seed),('계산 장치',self.device)]:form.addRow(title,w)
        av.addLayout(form);av.addWidget(label('결측은 자동 보간하지 않습니다. 센서별로 모든 비교 조건에 공통인 평가 시각을 사용합니다.','muted'));v.addWidget(advanced)
        self.training_summary=label('');v.addWidget(self.training_summary);v.addWidget(button('학습 시작 → 완료 후 결과 비교',self.start_training,True));v.addStretch()
        self.targets.itemChanged.connect(self.update_training_summary);self.env_mode.toggled.connect(self.update_training_summary)
        self.run_combo=QComboBox();self.run_combo.currentIndexChanged.connect(self.show_metrics);self.metrics_table=QTableView()
        # A short three-step flow keeps the next action visible without scrolling.
        host=self.stack.widget(2).widget();layout=host.layout()
        widgets=[layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget()]
        from PySide6.QtWidgets import QScrollArea
        steps=QTabWidget();steps.setMinimumHeight(220)
        for title,parts in [('1  센서 선택',[widgets[2]]),('2  모델 · 예측 기간',[widgets[3],widgets[4]]),('3  기간 확인 · 시작',[widgets[5],widgets[6]])]:
            panel=QWidget();pv=QVBoxLayout(panel)
            for part in parts:pv.addWidget(part)
            pv.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(panel);steps.addTab(scroll,title)
        layout.insertWidget(2,steps,1)
        start=widgets[-1];self.training_steps=steps
        controls=QHBoxLayout();back=button('← 이전',lambda:steps.setCurrentIndex(max(0,steps.currentIndex()-1)))
        forward=button('다음 →',lambda:steps.setCurrentIndex(min(2,steps.currentIndex()+1)),True)
        controls.addWidget(back);controls.addStretch();controls.addWidget(forward);layout.insertLayout(3,controls)
        def step_changed(i):back.setEnabled(i>0);forward.setVisible(i<2);start.setVisible(i==2)
        steps.currentChanged.connect(step_changed);step_changed(0)
        outer_scroll=self.stack.widget(2);outer_scroll.takeWidget();self.stack.removeWidget(outer_scroll);self.stack.insertWidget(2,host);outer_scroll.deleteLater()


    def select_sensors(self,checked):
        for i in range(self.targets.count()):self.targets.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    def period_changed(self):
        value=self.period.currentData();self.horizons.setVisible(value is None)
        if value:self.horizons.setText(value)
        self.update_training_summary()
    def update_training_summary(self,*args):
        if hasattr(self,'training_summary'):self.training_summary.setText(f'선택 센서 {len(self.checked(self.targets))}개 · 예측 {self.horizons.text()}시간 뒤 · 기존 결과는 보존됩니다.')

    def build_results(self):
        v=self.page('결과 비교','같은 센서·입력 조건·예측 기간의 최종 평가 결과를 비교합니다. MAE·RMSE는 낮을수록, R²는 높을수록 좋습니다.')
        row=QHBoxLayout();row.addWidget(label('학습 기록'));row.addWidget(self.run_combo,1);row.addWidget(button('전체 결과 CSV',self.export_metrics));v.addLayout(row)
        self.result_sensor=combo([('전체 센서','all')]);self.result_mode=combo([('센서 이력만','self'),('환경 포함','environment')]);self.result_horizon=QComboBox();self.result_split=combo([('최종 평가','test'),('모델 선택용 검증','validation')])
        row=QHBoxLayout()
        for w in (self.result_sensor,self.result_mode,self.result_horizon,self.result_split):row.addWidget(w);w.currentIndexChanged.connect(self.render_comparison)
        v.addLayout(row);self.result_summary=label('학습 기록을 선택하면 센서별 비교표가 표시됩니다.');v.addWidget(self.result_summary)
        self.show_baselines=QCheckBox('단순 기준 모델 지표도 표시');self.show_baselines.toggled.connect(self.render_comparison);v.addWidget(self.show_baselines)
        self.result_tabs=QTabWidget();self.result_tabs.addTab(self.metrics_table,'모델 비교표')
        plot=QWidget();pv=QVBoxLayout(plot);pv.addWidget(label('센서를 선택한 뒤 그래프를 불러오세요. 모든 센서를 선택하면 센서별 그래프를 만듭니다.','muted'));pv.addWidget(button('실제값 · 예측값 그래프 보기',self.comparison_plot,True))
        from PySide6.QtWidgets import QScrollArea
        scroll=QScrollArea();scroll.setWidgetResizable(True);self.compare_host=QWidget();self.compare_layout=QVBoxLayout(self.compare_host);scroll.setWidget(self.compare_host);pv.addWidget(scroll,1)
        self.result_tabs.addTab(plot,'실제값 · 예측값 그래프');v.addWidget(self.result_tabs,1)
        row=QHBoxLayout();row.addWidget(button('현재 비교표 저장',lambda:self.save_csv(self.metrics_table.model().df,'comparison.csv')));row.addWidget(button('논문 그림 만들기',lambda:self.navigate(4)));row.addStretch();v.addLayout(row)

    def show_metrics(self,*args):
        if not hasattr(self,'result_sensor'):return
        self.result_df=pd.DataFrame()
        if self.project and self.run_combo.currentData():
            r=self.project.get(self.run_combo.currentData());p=self.project.file(r['folder']+'/metrics.csv')
            if p.exists():self.result_df=pd.read_csv(p)
        for widget,column in [(self.result_sensor,'sensor'),(self.result_horizon,'horizon')]:
            previous=widget.currentData();widget.blockSignals(True);widget.clear()
            if column=='sensor':widget.addItem('전체 센서','all')
            if not self.result_df.empty:
                for x in sorted(self.result_df[column].unique()):widget.addItem(str(x)+('시간 뒤' if column=='horizon' else ''),int(x) if column=='horizon' else str(x))
            idx=widget.findData(previous);widget.setCurrentIndex(max(0,idx));widget.blockSignals(False)
        self.render_comparison()

    def comparison_data(self):
        df=getattr(self,'result_df',pd.DataFrame())
        if df.empty:return df
        out=df[(df['mode']==self.result_mode.currentData())&(df.horizon==self.result_horizon.currentData())&(df.split==self.result_split.currentData())]
        if self.result_sensor.currentData()!='all':out=out[out.sensor==self.result_sensor.currentData()]
        return out

    def render_comparison(self,*args):
        if not hasattr(self,'result_summary'):return
        df=self.comparison_data();rows=[];wins=losses=ties=0
        for sensor,g in df.groupby('sensor') if not df.empty else []:
            run=self.project.get(self.run_combo.currentData());meta=self.project.get(run['config']['dataset_id'])['columns'];row={'센서':sensor,'단위':meta.get(sensor,{}).get('unit','')}
            indexed=g.set_index('model');row['평가 수']=int(g.n.min())
            for key in (('lstm','transformer','persistence','linear') if self.show_baselines.isChecked() else ('lstm','transformer')):
                if key in indexed.index:
                    for metric in ('MAE','RMSE','R2'):row[MODEL[key]+'\n'+('R²' if metric=='R2' else metric)]=indexed.loc[key,metric]
            if {'lstm','transformer'}<=set(indexed.index):
                a=float(indexed.loc['lstm','MAE']);b=float(indexed.loc['transformer','MAE'])
                verdict='동일' if a==b else 'Transformer' if b<a else 'LSTM'
                row['MAE 우수 모델']=verdict;wins+=verdict=='Transformer';losses+=verdict=='LSTM';ties+=verdict=='동일'
                row['TF 오차 감소\n(%, 음수=악화)']=(a-b)/a*100 if a else np.nan
            rows.append(row)
        set_table(self.metrics_table,pd.DataFrame(rows))
        for col,name in enumerate(self.metrics_table.model().df.columns):
            self.metrics_table.setColumnWidth(col,60 if name in ('단위','평가 수') else 112)
        note=f'{self.result_split.currentText()} · Transformer MAE 우수 {wins}개 / LSTM 우수 {losses}개 / 동일 {ties}개. 단위가 다른 센서의 오차는 합산하지 않습니다.' if rows else '이 조건의 결과가 없습니다. 학습 기록·입력 조건·예측 기간을 확인하세요.'
        if self.project and self.run_combo.currentData() and affected(self.project,self.project.get(self.run_combo.currentData())['config']['dataset_id']):note='[취소된 자료를 포함한 과거 결과] '+note
        if rows and wins+losses+ties==0:note='LSTM과 Transformer가 함께 학습된 비교 결과가 없습니다. 현재 저장된 모델의 지표를 표시합니다.'
        self.result_summary.setText(note)
        if hasattr(self,'compare_layout'):
            while self.compare_layout.count():
                item=self.compare_layout.takeAt(0)
                if item.widget():item.widget().deleteLater()

    def comparison_plot(self):
        if not self.require_project() or not self.run_combo.currentData():return
        try:
            from .figures import make_figure,FigureSpec
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
            run=self.project.get(self.run_combo.currentData());pieces=[]
            for chunk in pd.read_csv(self.project.file(run['folder']+'/predictions.csv.gz'),chunksize=150000):
                g=chunk[(chunk['mode']==self.result_mode.currentData())&(chunk.horizon==self.result_horizon.currentData())&(chunk.split==self.result_split.currentData())]
                if self.result_sensor.currentData()!='all':g=g[g.sensor==self.result_sensor.currentData()]
                if not g.empty:pieces.append(g)
            if not pieces:raise ValueError('이 조건의 그래프 자료가 없습니다.')
            self.render_comparison()
            meta=self.project.get(run['config']['dataset_id'])['columns']
            for sensor,g in pd.concat(pieces).groupby('sensor'):
                g=g.copy();g['model']=g.model.map(MODEL);g['mode']=g['mode'].map(MODE)
                fig=make_figure(g,FigureSpec(kind='comparison',title=sensor,ylabel=meta[sensor]['unit'],xlabel='관측 시각'))
                canvas=FigureCanvasQTAgg(fig);canvas.setMinimumHeight(330);self.compare_layout.addWidget(NavigationToolbar2QT(canvas,self));self.compare_layout.addWidget(canvas);canvas.draw()
            self.result_tabs.setCurrentIndex(1)
        except Exception as e:self.error(e)

    def build_models(self):
        v=self.page('새 예측','① 센서와 모델 선택 → ② 예측 가능한 시각 확인 → ③ 예측 생성. 새 실제값이 들어오면 저장된 예측을 평가할 수 있습니다.')
        tabs=QTabWidget();v.addWidget(tabs,1);self.predict_tabs=tabs
        fresh=QWidget();fv=QVBoxLayout(fresh);row=QHBoxLayout();self.model_sensor=QComboBox();self.model_sensor.currentIndexChanged.connect(self.filter_models);row.addWidget(label('예측할 센서'));row.addWidget(self.model_sensor,1);fv.addLayout(row)
        self.models_table=QTableView();fv.addWidget(self.models_table,1);self.model_hint=label('표에서 사용할 모델을 선택하세요.','muted');fv.addWidget(self.model_hint)
        row=QHBoxLayout();row.addWidget(button('예측 가능한 시각 확인',self.check_origins,True));row.addWidget(button('이 센서의 기본 모델로 지정',self.assign_model));row.addStretch();fv.addLayout(row)
        self.origin_hint=label('모델을 선택한 후 위 버튼을 누르면 데이터의 빈 구간을 검사합니다.','muted');fv.addWidget(self.origin_hint)
        self.nearest_button=button('가장 최근의 예측 가능한 날짜 선택',self.select_latest_valid);self.nearest_button.setEnabled(False);fv.addWidget(self.nearest_button)
        row=QHBoxLayout();self.origin_mode=combo([('최신 관측 시점에서 예측','latest'),('달력에서 과거 시점 선택','calendar')]);self.origin_date=QDateEdit();self.origin_date.setCalendarPopup(True);self.origin_date.setDisplayFormat('yyyy-MM-dd');self.origin_time=QComboBox()
        for w in (self.origin_mode,self.origin_date,self.origin_time):row.addWidget(w)
        self.origin_mode.currentIndexChanged.connect(self.origin_changed);self.origin_date.dateChanged.connect(self.origin_times);self.origin_time.currentIndexChanged.connect(self.update_origin_hint);fv.addLayout(row)
        self.predict_button=button('예측 생성',self.predict_selected,True);self.predict_button.setEnabled(False);fv.addWidget(self.predict_button);tabs.addTab(fresh,'새 예측 만들기')
        saved=QWidget();sv=QVBoxLayout(saved);self.predictions_table=QTableView();sv.addWidget(self.predictions_table,1)
        row=QHBoxLayout();row.addWidget(button('예측값 · 그래프 보기',self.view_prediction));row.addWidget(button('예측 CSV 저장',self.export_prediction));row.addWidget(button('추가된 실제값으로 평가',self.evaluate_selected));sv.addLayout(row)
        self.evaluation_label=label('');sv.addWidget(self.evaluation_label);tabs.addTab(saved,'저장된 예측 · 사후 평가')

    def filter_models(self,*args):
        if not hasattr(self,'model_sensor'):return
        self.model_rows=[m for m in getattr(self,'all_models',[]) if m['sensor']==self.model_sensor.currentData()]
        rows=[]
        for m in self.model_rows:
            bad=affected(self.project,m['dataset_id']);rows.append({'기본':'●' if self.project.setting('default:'+m['sensor'])==m['id'] else '', '모델':MODEL.get(m['model'],m['model']),'입력':MODE.get(m['mode'],m['mode']),'예측 (시간 뒤)':', '.join(map(str,m['horizons'])),'검증 MAE':np.mean([x['MAE'] for x in m['metrics'] if x['split']=='validation']),'학습일':m['created'][:16].replace('T',' '),'상태':'자료 취소 · 재학습 필요' if bad else STATUS.get(m['status'],m['status'])})
        set_table(self.models_table,pd.DataFrame(rows));self.models_table.selectionModel().selectionChanged.connect(self.show_model_detail)
        self.valid_model=None;self.valid_times=pd.DatetimeIndex([]);self.predict_button.setEnabled(False)
        if rows:
            preferred=next((i for i,r in enumerate(rows) if r['기본']),0);self.models_table.selectRow(preferred)
        else:self.model_hint.setText('학습된 모델이 없습니다. 학습하기에서 모델을 먼저 만들어 주세요.')

    def show_model_detail(self,*args):
        if not hasattr(self,'predict_button'):return
        self.valid_model=None;self.predict_button.setEnabled(False)
        self.nearest_button.setEnabled(False)
        self.origin_date.setEnabled(False);self.origin_time.setEnabled(False)
        try:
            m=self.selected(self.models_table,self.model_rows)
            self.model_hint.setText(f"{m['sensor']} · {MODEL.get(m['model'],m['model'])} · 과거 {m['lookback']}시간 참고 → {', '.join(map(str,m['horizons']))}시간 뒤 예측 · 단위 {m['columns_meta'][m['sensor']]['unit']}")
            self.origin_hint.setText('예측 가능한 시각 확인을 눌러 이 모델의 입력 데이터를 검사하세요.')
        except ValueError:pass

    def check_origins(self):
        if not self.require_project() or not self.idle():return
        try:
            m=self.selected(self.models_table,self.model_rows)
            if m['status']!='complete' or self.project.get(m['run_id'])['status']!='complete':raise ValueError('완료된 학습 모델을 선택하세요.')
            self.start_job('origins',{'model_id':m['id']})
        except Exception as e:self.error(e)

    def origin_changed(self,*args):
        historical=self.origin_mode.currentData()=='calendar' and self.valid_model is not None;self.origin_date.setEnabled(historical);self.origin_time.setEnabled(historical);self.update_origin_hint()
    def select_latest_valid(self):
        if not self.valid_model or not len(self.valid_times):return
        self.origin_mode.setCurrentIndex(1);self.origin_date.setDate(QDate.fromString(str(self.valid_times[-1].date()),'yyyy-MM-dd'));self.origin_times()
    def origin_times(self,*args):
        self.origin_time.blockSignals(True);self.origin_time.clear();day=self.origin_date.date().toString('yyyy-MM-dd')
        for t in self.valid_times[self.valid_times.strftime('%Y-%m-%d')==day]:self.origin_time.addItem(t.strftime('%H:%M'),str(t))
        self.origin_time.setCurrentIndex(self.origin_time.count()-1);self.origin_time.blockSignals(False);self.update_origin_hint()
    def chosen_origin(self):
        if not self.valid_model:return None
        if self.origin_mode.currentData()=='latest':return self.latest_origin if pd.Timestamp(self.latest_origin) in self.valid_times else None
        return self.origin_time.currentData()
    def update_origin_hint(self,*args):
        if not hasattr(self,'predict_button'):return
        chosen=self.chosen_origin();self.predict_button.setEnabled(bool(chosen) and not self.busy())
        if not self.valid_model:return
        if chosen:
            m=self.project.get(self.valid_model);end=pd.Timestamp(chosen)+pd.Timedelta(hours=max(m['horizons']))
            old=pd.Timestamp(chosen)<pd.Timestamp(self.latest_origin)
            self.origin_hint.setText(f"{'과거 시점 재현' if old else '최신 관측 이후 예측'}: {chosen} → {end}. {'현재 이후의 미래 예측이 아닙니다.' if old else ''}")
        else:self.origin_hint.setText('이 시점에는 필요한 연속 관측이 부족합니다. 달력에서 다른 날짜를 고르거나 새 데이터를 추가하세요. 과거로 자동 변경하지 않습니다.')
    def predict_selected(self):
        if not self.idle():return
        try:
            m=self.selected(self.models_table,self.model_rows);origin=self.chosen_origin()
            if m['id']!=self.valid_model or not origin:raise ValueError('예측 가능한 시각을 먼저 확인하고 선택하세요.')
            self.start_job('predict',{'model_id':m['id'],'origin':origin})
        except Exception as e:self.error(e)

    def view_prediction(self):
        try:
            p=self.selected(self.predictions_table,self.pred_rows);df=pd.read_csv(self.project.file(p['file']));dialog=QDialog(self);dialog.setWindowTitle('저장된 예측 · '+p['sensor']);dialog.resize(1000,750);v=QVBoxLayout(dialog)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            fig=Figure(layout='constrained');ax=fig.add_subplot();ax.plot(pd.to_datetime(df.time),df.predicted,'o-',label='Prediction');ax.set_ylabel(p['unit']);ax.legend();canvas=FigureCanvasQTAgg(fig);v.addWidget(canvas,1)
            values=df[['time','horizon','predicted']].rename(columns={'time':'예측 대상 시각','horizon':'몇 시간 뒤','predicted':'예측값 ('+p['unit']+')'});table=QTableView();set_table(table,values);v.addWidget(table,1);dialog.exec()
        except Exception as e:self.error(e)

    def change_import(self,enabled):
        if not self.require_project() or not self.idle():return
        try:
            r=self.selected(self.import_table,self.import_rows)
            if (r['id'] not in disabled(self.project))==enabled:raise ValueError('이미 '+('등록된' if enabled else '취소된')+' 자료입니다.')
            from .history import sources
            count=sum(r['id'] in sources(self.project,m['dataset_id']) for m in self.project.list('model'))
            action='복원' if enabled else '취소'
            if QMessageBox.question(self,'등록 '+action,f"{r['source_name']} 등록을 {action}할까요?\n관련 학습 모델 {count}개. 기존 모델·예측은 보존하고 자료 상태를 표시합니다.\n원래 엑셀 파일은 삭제하지 않습니다.")!=QMessageBox.StandardButton.Yes:return
            self.start_job('registration',{'id':r['id'],'enabled':enabled})
        except Exception as e:self.error(e)

    def refresh(self):
        previous_dataset=getattr(self,'form_dataset',None)
        super().refresh()
        if not hasattr(self,'import_table'):return
        latest=self.project.latest_dataset() if self.project else None
        if latest:
            idx=self.dataset_combo.findData(latest['id']);self.dataset_combo.setCurrentIndex(idx)
        elif self.project:
            self.data_summary.setText('현재 사용할 데이터가 없습니다. 등록 내역에서 복원하거나 새 파일을 추가하세요. 아래 버전은 과거 기록입니다.')
        self.import_rows=registrations(self.project) if self.project else [];excluded=disabled(self.project) if self.project else set()
        set_table(self.import_table,pd.DataFrame([{'상태':'취소됨' if r['id'] in excluded else '등록됨','파일':r['source_name'],'시트':', '.join(m['sheet'] for m in r.get('mappings',[])),'등록일':r['created'][:19].replace('T',' '),'기간':r['start'][:10]+' ~ '+r['end'][:10]} for r in self.import_rows]))
        self.all_models=self.project.list('model') if self.project else [];previous=self.model_sensor.currentData();self.model_sensor.blockSignals(True);self.model_sensor.clear()
        for sensor in sorted({m['sensor'] for m in self.all_models}):self.model_sensor.addItem(sensor,sensor)
        self.model_sensor.setCurrentIndex(max(0,self.model_sensor.findData(previous)));self.model_sensor.blockSignals(False);self.filter_models()
        if self.pred_rows:
            set_table(self.predictions_table,pd.DataFrame([{'센서':p['sensor'],'예측 기준':p['origin'],'예측 끝':p['last_target'],'생성일':p['created'][:19].replace('T',' '),'자료 상태':'취소된 자료 포함' if affected(self.project,p['dataset_id']) or affected(self.project,self.project.get(p['model_id'])['dataset_id']) else '정상'} for p in self.pred_rows]));self.predictions_table.selectRow(0)
        for i in range(self.run_combo.count()):
            r=self.project.get(self.run_combo.itemData(i));self.run_combo.setItemText(i,f"{r['created'][:16].replace('T',' ')} · {STATUS.get(r['status'],r['status'])} · 모델 {len(r['models'])}개")
        if self.form_dataset and self.form_dataset!=previous_dataset:self.select_sensors(True)
        self.update_training_summary()
        if self.project and self.project.list('evaluation'):
            self.show_evaluation(self.project.list('evaluation')[0])
        if hasattr(self,'fig_start') and isinstance(self.fig_start,OptionalDate) and latest:
            self.fig_start.date.setText(latest['start']);self.fig_end.date.setText(latest['end'])

    def start_job(self,kind,config):
        self.active_kind=kind;super().start_job(kind,config);self.progress.show();self.predict_button.setEnabled(False)
        if kind=='origins':self.models_table.setEnabled(False);self.model_sensor.setEnabled(False)

    def job_finished(self,*args):
        kind=getattr(self,'active_kind','');resultpath=self.job_path.with_suffix('.result.json');result=read_json(resultpath) if resultpath.exists() else {}
        super().job_finished(*args);self.progress.hide()
        self.models_table.setEnabled(True);self.model_sensor.setEnabled(True)
        if not result.get('ok'):return
        if kind=='origins':
            data=result['id'];self.valid_model=data['model_id'];self.valid_times=pd.DatetimeIndex(pd.to_datetime(data['times']));self.latest_origin=data['latest']
            index=next((i for i,m in enumerate(self.model_rows) if m['id']==self.valid_model),None)
            if index is not None:
                self.models_table.selectRow(index);self.valid_model=data['model_id']
            if len(self.valid_times):
                self.origin_date.setMinimumDate(QDate.fromString(str(self.valid_times.min().date()),'yyyy-MM-dd'));self.origin_date.setMaximumDate(QDate.fromString(str(self.valid_times.max().date()),'yyyy-MM-dd'));self.origin_date.setDate(self.origin_date.maximumDate())
            self.origin_times();self.origin_changed()
            self.nearest_button.setEnabled(bool(len(self.valid_times)))
        elif kind=='train':self.navigate(5)
        elif kind=='predict':self.predict_tabs.setCurrentIndex(1);self.navigate(3)
        elif kind=='evaluate':
            self.show_evaluation(self.project.get(result['id']))

    def show_evaluation(self,e):
        m=e['metrics']
        if not m.get('n'):
            self.evaluation_label.setText('아직 비교할 실제 관측값이 없습니다. 예측한 날짜의 데이터를 추가한 뒤 다시 평가하세요.')
        else:
            fields=[('관측 수','n'),('MAE','MAE'),('RMSE','RMSE'),('R²','R2')]
            self.evaluation_label.setText('최근 사후 평가 · '+' / '.join(f"{name}: {('계산 불가' if m.get(key) is None else format(m[key],'.8g'))}" for name,key in fields))

    def assign_model(self):
        try:
            m=self.selected(self.models_table,self.model_rows)
            if affected(self.project,m['dataset_id']):raise ValueError('취소한 데이터로 학습한 모델입니다. 현재 데이터로 다시 학습하세요.')
        except Exception as e:return self.error(e)
        super().assign_model()

    def build_figures(self):
        super().build_figures()
        page=self.stack.widget(4).widget();v=page.layout();old=v.itemAt(2).widget()
        # Keep every publication control, moving typography and sizing into an optional section.
        settings=QWidget();sv=QVBoxLayout(settings);form=QFormLayout()
        for title,w in [('그림 종류',self.fig_kind),('센서',self.fig_sensor),('학습 기록',self.fig_run),('예측 기간 (시간 뒤)',self.fig_horizon)]:form.addRow(title,w)
        sv.addLayout(form);extra,ev=fold('크기 · 글꼴 · 기간 등 상세 설정');grid=QFormLayout()
        self.fig_start=OptionalDate();self.fig_end=OptionalDate()
        for title,w in [('산점도·잔차의 모델',self.fig_model),('너비 (mm)',self.fig_width),('높이 (mm)',self.fig_height),('해상도',self.fig_dpi),('글꼴',self.fig_font),('글자 크기',self.fig_size),('시작',self.fig_start),('종료',self.fig_end),('제목',self.fig_title),('가로축',self.fig_xlabel),('세로축',self.fig_ylabel)]:grid.addRow(title,w)
        ev.addLayout(grid);ev.addWidget(self.fig_gray);ev.addWidget(self.fig_legend);sv.addWidget(extra)
        row=QHBoxLayout();row.addWidget(button('미리보기',self.preview_figure));row.addWidget(button('논문 그림 저장',self.save_figure,True));sv.addLayout(row);v.replaceWidget(old,settings);old.hide();old.deleteLater()

    def open_manual(self):
        QMessageBox.information(self,'사용 순서','1. 고분별 프로젝트를 만드세요.\n2. 데이터에서 엑셀·CSV를 추가하고 날짜·센서·단위를 확인하세요. 잘못 올렸으면 등록 내역에서 취소하세요.\n3. 학습하기에서 센서·예측 기간·모델을 고르세요.\n4. 결과 비교에서 수치와 실제값·예측값 그래프를 확인하세요.\n5. 새 예측에서 모델과 유효한 시각을 선택하세요.\n6. 새 실제값을 추가하면 저장된 예측의 사후 평가가 가능합니다.\n7. 논문 그림에서 규격을 정해 PDF·SVG·PNG·TIFF로 저장하세요.\n\n프로젝트 내보내기로 데이터·모델·결과를 백업할 수 있습니다.')
