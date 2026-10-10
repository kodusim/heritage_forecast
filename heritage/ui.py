from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt,QProcess,QTimer,QThread,Signal,QAbstractTableModel
from PySide6.QtGui import QColor,QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QFormLayout,
    QLabel,QPushButton,QComboBox,QStackedWidget,QFrame,QScrollArea,QLineEdit,QSpinBox,QDoubleSpinBox,
    QCheckBox,QListWidget,QListWidgetItem,QTableView,QTableWidget,QTableWidgetItem,QHeaderView,
    QFileDialog,QMessageBox,QInputDialog,QDialog,QDialogButtonBox,QPlainTextEdit,QProgressBar,QSplitter,QTextBrowser)
from .storage import Project,projects,root_dir,uid,write_json,read_json

STYLE='''
QMainWindow,QDialog { background:#f4f6f9; }
QWidget { font-family:"Malgun Gothic"; font-size:13px; color:#243247; }
QFrame#sidebar { background:#142638; border:0; }
QFrame#sidebar QLabel { color:#bac8d6; }
QLabel#brand { color:white; font-size:24px; font-weight:700; }
QLabel#pageTitle { font-size:25px; font-weight:700; color:#142638; }
QLabel#muted { color:#64748b; }
QLabel#cardValue { font-size:23px; font-weight:700; color:#0b806e; }
QFrame#card { background:white; border:1px solid #e0e6ee; border-radius:10px; }
QPushButton { background:white; border:1px solid #ccd6e1; padding:9px 16px; border-radius:6px; font-weight:600; }
QPushButton:hover { background:#eaf1f7; border-color:#9cabbc; }
QPushButton:disabled { color:#9da8b5; background:#f0f2f5; border-color:#e1e5eb; }
QPushButton#primary { background:#087f70; border-color:#087f70; color:white; }
QPushButton#primary:hover { background:#066759; }
QPushButton#nav { text-align:left; padding:13px 16px; background:transparent; border:0; color:#c5d2df; font-size:14px; }
QPushButton#nav:checked { color:white; background:#264258; border-left:3px solid #44c6b1; border-radius:4px; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:white; border:1px solid #ccd6e1; border-radius:5px; padding:7px; min-height:20px; }
QComboBox::drop-down { border:0; width:22px; }
QTableView,QTableWidget,QListWidget,QPlainTextEdit { background:white; border:1px solid #dfe5ed; border-radius:6px; alternate-background-color:#f7f9fc; }
QHeaderView::section { background:#edf2f7; border:0; border-bottom:1px solid #d9e2ec; padding:9px; font-weight:600; }
QTableView::item { padding:6px; } QTableView::item:selected,QListWidget::item:selected {background:#dbf1ed;color:#173d39;}
QListWidget::item {padding:7px;}
QScrollArea { border:0; background:transparent; }
QProgressBar { border:0; border-radius:3px; background:#e0e7ef; height:7px; text-align:center; }
QProgressBar::chunk {background:#10a58e;border-radius:3px;}
QCheckBox {spacing:8px;padding:3px;} QToolTip {background:white;color:#243247;border:1px solid #ccd6e1;padding:6px;}
'''

def label(text,object_name=None,wrap=True):
    w=QLabel(text);w.setWordWrap(wrap)
    if object_name:w.setObjectName(object_name)
    return w

def button(text,callback,primary=False):
    b=QPushButton(text);b.clicked.connect(callback)
    if primary:b.setObjectName('primary')
    return b

def card(title):
    w=QFrame();w.setObjectName('card');v=QVBoxLayout(w);v.setContentsMargins(20,18,20,18);v.setSpacing(12)
    if title:v.addWidget(label(title))
    return w,v

def combo(items):
    c=QComboBox()
    for text,value in items:c.addItem(text,value)
    return c

class FrameModel(QAbstractTableModel):
    def __init__(self,df,parent=None):super().__init__(parent);self.df=df
    def rowCount(self,parent=None):return len(self.df)
    def columnCount(self,parent=None):return len(self.df.columns)
    def data(self,index,role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():return None
        if role==Qt.ItemDataRole.DisplayRole:
            v=self.df.iloc[index.row(),index.column()]
            if pd.isna(v):return '—'
            if isinstance(v,(float,np.floating)):return f'{v:.6g}'
            return str(v)
        return None
    def headerData(self,section,orientation,role=Qt.ItemDataRole.DisplayRole):
        if role!=Qt.ItemDataRole.DisplayRole:return None
        return str(self.df.columns[section]) if orientation==Qt.Orientation.Horizontal else str(section+1)

def set_table(view,df):
    view.setModel(FrameModel(df,view));view.setAlternatingRowColors(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    view.horizontalHeader().setStretchLastSection(True);view.verticalHeader().hide()
    view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows);view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
    view.resizeColumnsToContents()
    for i in range(len(df.columns)):view.setColumnWidth(i,min(300,max(105,view.columnWidth(i))))

class InspectThread(QThread):
    ready=Signal(object);failed=Signal(str)
    def __init__(self,path):super().__init__();self.path=path
    def run(self):
        try:
            from .data import inspect_file
            self.ready.emit(inspect_file(self.path))
        except Exception as e:self.failed.emit(str(e))

class ImportDialog(QDialog):
    def __init__(self,path,parent):
        super().__init__(parent);self.path=path;self.sheets=[];self.req=None
        self.setWindowTitle('데이터 등록 · 열과 센서 확인');self.resize(1000,760)
        v=QVBoxLayout(self);v.setContentsMargins(24,24,24,24);v.setSpacing(14)
        v.addWidget(label('데이터를 확인하고 등록하세요','pageTitle'))
        v.addWidget(label(Path(path).name+'\n파일명 대신 실제 날짜·센서 열을 확인합니다. 미리보기는 앞부분만 보여줍니다.','muted'))
        row=QHBoxLayout();self.sheet=QComboBox();self.header=QSpinBox();self.header.setRange(1,20);self.datecol=QSpinBox();self.datecol.setRange(1,100)
        row.addWidget(label('시트'));row.addWidget(self.sheet,2);row.addWidget(label('헤더 행'));row.addWidget(self.header);row.addWidget(label('날짜 열 번호'));row.addWidget(self.datecol);row.addWidget(button('매핑 다시 읽기',self.populate));v.addLayout(row)
        self.preview=QTableView();self.preview.setMaximumHeight(190);v.addWidget(self.preview)
        self.mapping=QTableWidget(0,6);self.mapping.setHorizontalHeaderLabels(['사용','원본 열','센서 이름','역할','단위','시간 집계']);self.mapping.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch);self.mapping.verticalHeader().hide();v.addWidget(self.mapping,1)
        self.policy=combo([('충돌 시 등록 중지 (기본)','reject'),('기존 값 유지','old'),('새 파일 값 사용','new'),('충돌한 값은 결측으로 제외','mask')])
        form=QFormLayout();form.addRow('같은 시간·센서의 값이 다를 때',self.policy);v.addLayout(form)
        self.status=label('엑셀의 시트와 헤더를 읽는 중입니다…','muted');v.addWidget(self.status)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);bb.button(QDialogButtonBox.StandardButton.Ok).setText('검사 후 새 버전 등록');bb.button(QDialogButtonBox.StandardButton.Cancel).setText('닫기');bb.accepted.connect(self.submit);bb.rejected.connect(self.reject);v.addWidget(bb);self.bb=bb;bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.sheet.currentIndexChanged.connect(self.select_sheet)
        self.thread=InspectThread(path);self.thread.ready.connect(self.loaded);self.thread.failed.connect(self.failed);self.thread.start()
    def failed(self,msg):self.status.setText(msg)
    def loaded(self,sheets):
        self.sheets=sheets;self.sheet.blockSignals(True)
        for s in sheets:self.sheet.addItem(s['name'])
        # Prefer a consolidated sheet for initial inspection; never concatenate it with annual tabs.
        selected=next((i for i,s in enumerate(sheets) if '합본' in s['name']),0)
        self.sheet.setCurrentIndex(selected);self.sheet.blockSignals(False);self.select_sheet(selected)
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
    def select_sheet(self,i):
        if not self.sheets or i<0:return
        from .data import infer_mapping
        rows=self.sheets[i]['rows'];set_table(self.preview,pd.DataFrame(rows).rename(columns=lambda c:f'{c+1}열'))
        try:h,d=infer_mapping(rows);self.header.setValue(h+1);self.datecol.setValue(d+1);self.status.setText('자동 감지한 열을 확인하세요. 연도별 환경 시트는 위치·단위가 다를 수 있습니다.')
        except ValueError as e:self.status.setText(str(e))
        self.populate()
    def populate(self):
        if not self.sheets:return
        from .data import mapping_names
        import re
        rows=self.sheets[self.sheet.currentIndex()]['rows'];h=self.header.value()-1;d=self.datecol.value()-1
        if h>=len(rows):return
        names=mapping_names(rows,h,d);self.mapping.setRowCount(len(names))
        for r,(c,n) in enumerate(names.items()):
            use=QCheckBox();known=bool(re.search(r'([DT]\d|Temp|AirTC|Rain|Wind|HMP|RH|Soil|기온|습도|강수|T\d|D\d|C\[)',n,re.I))
            numeric=pd.to_numeric(pd.Series([row[c] if c<len(row) else None for row in rows[h+1:]]),errors='coerce').notna().any()
            use.setChecked(bool(known and numeric));self.mapping.setCellWidget(r,0,use)
            item=QTableWidgetItem(str(c+1));item.setFlags(item.flags()&~Qt.ItemFlag.ItemIsEditable);self.mapping.setItem(r,1,item);self.mapping.setItem(r,2,QTableWidgetItem(n))
            environmental=bool(re.search(r'Temp|AirTC|Rain|Wind|HMP|RH|Soil|기온|습도|강수|C\[',n,re.I))
            role=combo([('거동 · 예측 대상','target'),('환경 · 입력','environment')]);role.setCurrentIndex(1 if environmental else 0);self.mapping.setCellWidget(r,3,role)
            unit='°C' if re.search(r'Temp|AirTC|기온|C\[',n,re.I) else '%' if re.search('RH|습도',n,re.I) else 'mm' if re.search(r'Rain|강수|D\d',n,re.I) else '°' if re.search(r'Dir|WD|T\d',n,re.I) else 'm/s' if 'speed' in n.lower() else ''
            self.mapping.setItem(r,4,QTableWidgetItem(unit))
            aggregation=combo([('평균','mean'),('마지막 값','last'),('합계','sum'),('풍향 원형 평균','circular')])
            method='last' if re.search('Rain_day|일누적',n,re.I) else 'sum' if re.search('Rain_Tot',n,re.I) else 'circular' if re.search('Wind.*Dir|WD_|풍향',n,re.I) else 'mean'
            aggregation.setCurrentIndex(aggregation.findData(method));self.mapping.setCellWidget(r,5,aggregation)
    def submit(self):
        cols={};meta={}
        for r in range(self.mapping.rowCount()):
            if not self.mapping.cellWidget(r,0).isChecked():continue
            n=self.mapping.item(r,2).text().strip();unit=self.mapping.item(r,4).text().strip()
            if not n or n in meta:return QMessageBox.warning(self,'열 매핑','센서 이름은 비어 있거나 중복될 수 없습니다.')
            if not unit:return QMessageBox.warning(self,'단위 확인',f'{n}의 단위를 입력하세요. 알 수 없는 경우 unknown으로 명시하세요.')
            cols[str(int(self.mapping.item(r,1).text())-1)]=n
            meta[n]={'role':self.mapping.cellWidget(r,3).currentData(),'unit':unit,'aggregation':self.mapping.cellWidget(r,5).currentData()}
        if not cols:return QMessageBox.warning(self,'열 매핑','사용할 수치 열을 선택하세요.')
        self.req={'path':self.path,'mappings':[{'sheet':self.sheet.currentText(),'header':self.header.value()-1,'time_col':self.datecol.value()-1,'columns':cols}],
                  'columns_meta':meta,'conflicts':self.policy.currentData()};self.accept()
    def done(self,result):
        if self.thread.isRunning():
            self.status.setText('헤더 검사를 마친 뒤 창을 닫을 수 있습니다.');return
        super().done(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Heritage Studio · 문화유산 시계열 연구');self.resize(1360,900);self.setMinimumSize(1080,740);self.setStyleSheet(STYLE)
        self.project=None;self.process=None;self.job_path=None;self.buffer='';self.model_rows=[];self.pred_rows=[];self.figure_canvas=None;self.figure_data=None;self.current_figure_spec=None;self.form_dataset=None
        central=QWidget();self.setCentralWidget(central);outer=QHBoxLayout(central);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        sidebar=QFrame();sidebar.setObjectName('sidebar');sidebar.setFixedWidth(230);left=QVBoxLayout(sidebar);left.setContentsMargins(20,30,20,24);left.setSpacing(10)
        left.addWidget(label('Heritage\nStudio','brand'));left.addWidget(label('문화유산 시계열 연구실'));left.addSpacing(28)
        self.nav=[];self.stack=QStackedWidget()
        for i,text in enumerate(['01   프로젝트 현황','02   데이터 관리','03   학습과 비교','04   모델과 예측','05   논문 Figure']):
            b=button(text,lambda checked=False,n=i:self.navigate(n));b.setObjectName('nav');b.setCheckable(True);left.addWidget(b);self.nav.append(b)
        left.addWidget(button('Transformer 결과',self.open_review,True))
        left.addStretch();left.addWidget(button('사용 설명서',self.open_manual));left.addWidget(label('LOCAL WORKSPACE\n데이터는 이 PC에 저장됩니다.'));left.addWidget(label('v1.0 · 오프라인 연구 도구'))
        outer.addWidget(sidebar)
        content=QWidget();right=QVBoxLayout(content);right.setContentsMargins(28,22,28,18);right.setSpacing(15)
        top=QHBoxLayout();top.addWidget(label('프로젝트','muted'));self.project_combo=QComboBox();self.project_combo.setMinimumWidth(210);top.addWidget(self.project_combo,1)
        top.addWidget(button('+ 새 프로젝트',self.new_project));top.addWidget(button('백업 복원',self.restore_project));top.addWidget(button('프로젝트 내보내기',self.export_project));right.addLayout(top);right.addWidget(self.stack,1)
        self.build_home();self.build_data();self.build_training();self.build_models();self.build_figures()
        job,jv=card('');job.setMaximumHeight(146);self.job_label=label('준비됨 · 데이터 등록부터 시작하세요.');self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);self.progress.setTextVisible(False)
        jobrow=QHBoxLayout();jobrow.addWidget(self.job_label,1);self.cancel_button=button('작업 취소',self.cancel_job);self.cancel_button.setEnabled(False);jobrow.addWidget(self.cancel_button);jv.addLayout(jobrow);jv.addWidget(self.progress)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(50);self.log.setMaximumBlockCount(500);jv.addWidget(self.log);right.addWidget(job)
        outer.addWidget(content,1);self.project_combo.currentIndexChanged.connect(self.select_project);self.refresh_projects();self.navigate(0)
        self.event_timer=QTimer(self);self.event_timer.setInterval(400);self.event_timer.timeout.connect(self.read_events);self.event_offset=0

    def page(self,title,description):
        scroll=QScrollArea();scroll.setWidgetResizable(True);w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(0,0,6,4);v.setSpacing(16)
        v.addWidget(label(title,'pageTitle'));v.addWidget(label(description,'muted'));scroll.setWidget(w);self.stack.addWidget(scroll);return v

    def build_home(self):
        v=self.page('자료에서 예측까지, 한 곳에서','데이터를 확인하고 같은 조건으로 모델을 비교하세요. 모든 실행과 결과를 프로젝트에 보관합니다.')
        grid=QGridLayout();self.home_values=[]
        for i,t in enumerate(['데이터 버전','거동 센서','완료 모델','저장된 예측']):
            c,cv=card(t);value=label('—','cardValue');cv.addWidget(value);grid.addWidget(c,0,i);self.home_values.append(value)
        v.addLayout(grid)
        c,cv=card('처음 사용하시나요?');
        for title,desc,page in [('1. 데이터 등록','엑셀·CSV의 시트, 날짜 열, 센서와 단위를 확인합니다.',1),('2. 학습 비교','기간과 센서를 선택해 기준선·LSTM·Transformer를 비교합니다.',2),('3. 예측과 그림 저장','저장 모델로 예측하고 논문 규격의 Figure를 만듭니다.',3)]:
            row=QHBoxLayout();text=QVBoxLayout();text.addWidget(label(title));text.addWidget(label(desc,'muted'));row.addLayout(text,1);row.addWidget(button('열기 →',lambda checked=False,n=page:self.navigate(n)));cv.addLayout(row)
        v.addWidget(c);c,cv=card('현재 프로젝트');self.home_detail=label('프로젝트를 만들거나 백업 파일을 복원하세요.','muted');cv.addWidget(self.home_detail);v.addWidget(c)
        row=QHBoxLayout();row.addWidget(button('연습용 예제 프로젝트 만들기',self.demo_project));row.addStretch();v.addLayout(row);v.addStretch()

    def build_data(self):
        v=self.page('데이터 관리','새 파일은 기존 자료와 대조하여 새 버전으로 저장합니다. 원본과 이전 결과는 유지됩니다.')
        row=QHBoxLayout();row.addWidget(button('+ 엑셀 / CSV 추가',self.import_file,True));row.addWidget(label('데이터 버전','muted'));self.dataset_combo=QComboBox();row.addWidget(self.dataset_combo,1);row.addWidget(button('품질표 내보내기',self.export_quality));v.addLayout(row)
        self.data_summary=label('자료를 추가하면 센서별 관측 기간과 결측률을 확인할 수 있습니다.','muted');v.addWidget(self.data_summary)
        self.quality_table=QTableView();self.quality_table.setMinimumHeight(270);v.addWidget(self.quality_table,1)
        c,cv=card('등록 안내');cv.addWidget(label('• 거동과 환경 파일을 같은 프로젝트에 차례로 등록하면 시각으로 결합됩니다.\n• 연도별 시트는 하나씩 등록하세요. 합본과 연도별 시트를 무조건 합치지 마세요.\n• 센서 위치·단위·영점이 바뀌면 새 센서 이름을 사용하세요.\n• 표의 결측률은 등록된 시각 기준입니다. 학습 시에는 규칙적인 시간격자로 다시 검사합니다.','muted'));v.addWidget(c)
        self.dataset_combo.currentIndexChanged.connect(self.show_quality)

    def build_training(self):
        v=self.page('학습과 비교','자기 센서 과거값만 사용하는 조건과 환경을 포함한 조건을 같은 테스트 시각에서 비교합니다.')
        top=QHBoxLayout();a,av=card('1 · 예측할 센서');self.targets=QListWidget();self.targets.setFixedHeight(110);av.addWidget(self.targets);b,bv=card('2 · 환경 입력');self.envs=QListWidget();self.envs.setFixedHeight(110);bv.addWidget(self.envs);top.addWidget(a,1);top.addWidget(b,1);v.addLayout(top)
        c,cv=card('3 · 학습 조건');form=QGridLayout();self.train_end=QLineEdit();self.val_end=QLineEdit();self.lookback=QSpinBox();self.lookback.setRange(2,720);self.lookback.setValue(24)
        self.horizons=QLineEdit('1, 6, 24');self.epochs=QSpinBox();self.epochs.setRange(1,500);self.epochs.setValue(20);self.minimum=QSpinBox();self.minimum.setRange(1,60);self.minimum.setValue(4)
        self.seed=QSpinBox();self.seed.setRange(0,100000);self.seed.setValue(42);self.device=combo([('CPU (기본)','cpu'),('NVIDIA CUDA','cuda')])
        fields=[('학습 종료 시각',self.train_end),('검증 종료 시각',self.val_end),('과거 입력 길이 · 시간',self.lookback),('예측 간격 · 시간',self.horizons),('최대 epoch',self.epochs),('시간당 최소 관측 수',self.minimum),('랜덤 시드',self.seed),('계산 장치',self.device)]
        for i,(t,w) in enumerate(fields):form.addWidget(label(t,'muted'),(i//4)*2,i%4);form.addWidget(w,(i//4)*2+1,i%4)
        for col in range(4):form.setColumnStretch(col,1)
        form.setHorizontalSpacing(24);form.setVerticalSpacing(5);cv.addLayout(form)
        self.model_checks={};mr=QHBoxLayout()
        for key,text in [('linear','선형 기준선'),('lstm','LSTM'),('transformer','Transformer')]:q=QCheckBox(text);q.setChecked(True);mr.addWidget(q);self.model_checks[key]=q
        mr.addWidget(label('Persistence는 항상 비교합니다.','muted'));cv.addLayout(mr)
        self.self_mode=QCheckBox('자기 과거 입력');self.self_mode.setChecked(True);self.env_mode=QCheckBox('환경 포함 입력');rr=QHBoxLayout();rr.addWidget(self.self_mode);rr.addWidget(self.env_mode);rr.addStretch();cv.addLayout(rr)
        cv.addWidget(label('1시간 단위로 센서별 집계 설정을 적용하며 결측을 보간하지 않습니다. 검증 종료 이후가 테스트입니다. 여러 예측 간격에 실제값이 모두 있는 공통 창만 비교합니다.','muted'))
        cv.addWidget(button('선택한 조건으로 학습 시작',self.start_training,True));v.addWidget(c)
        self.run_combo=QComboBox();self.run_combo.currentIndexChanged.connect(self.show_metrics);rr=QHBoxLayout();rr.addWidget(label('실행 결과'));rr.addWidget(self.run_combo,1);rr.addWidget(button('결과 CSV 내보내기',self.export_metrics));v.addLayout(rr)
        self.metrics_table=QTableView();self.metrics_table.setMinimumHeight(220);v.addWidget(self.metrics_table)

    def build_models(self):
        v=self.page('모델과 예측','학습 완료와 사용 모델 선택은 별개입니다. 미래 예측은 마지막 관측 시각을 기준으로 생성합니다.')
        self.models_table=QTableView();self.models_table.setMinimumHeight(220);v.addWidget(self.models_table)
        rr=QHBoxLayout();rr.addWidget(label('예측 기준 시각'));self.pred_origin=QLineEdit();self.pred_origin.setPlaceholderText('비우면 최신 · 직접 지정 예: 2025-01-01 00:00:00');rr.addWidget(self.pred_origin,1);v.addLayout(rr)
        r=QHBoxLayout();r.addWidget(button('선택 모델을 기본으로 지정',self.assign_model));r.addWidget(button('선택 모델로 새 예측',self.predict_selected,True));r.addStretch();v.addLayout(r)
        self.model_hint=label('모델을 선택하면 입력 조건과 검증 상태를 확인할 수 있습니다.','muted');v.addWidget(self.model_hint)
        c,cv=card('저장된 미래 예측');self.predictions_table=QTableView();self.predictions_table.setMinimumHeight(180);cv.addWidget(self.predictions_table)
        r=QHBoxLayout();r.addWidget(button('예측 CSV 내보내기',self.export_prediction));r.addWidget(button('현재 실제값으로 사후 평가',self.evaluate_selected));cv.addLayout(r);v.addWidget(c)
        self.evaluation_label=label('예측 생성 후 실제값이 추가되면 사후 평가를 실행할 수 있습니다.','muted');v.addWidget(self.evaluation_label);v.addStretch()

    def build_figures(self):
        v=self.page('논문 Figure','저널 규격에 맞는 벡터 그림과 고해상도 이미지를 만드세요. 사용 데이터와 설정이 함께 저장됩니다.')
        c,cv=card('그림 설정');grid=QGridLayout()
        self.fig_kind=combo([('관측 시계열','series'),('실제값과 모델 예측 비교','comparison'),('실제값 · 예측값 산점도','scatter'),('잔차 시계열','residual'),('예측 간격별 RMSE','horizon'),('센서·환경 상관행렬','correlation'),('월별 결측률','missing')])
        self.fig_sensor=QComboBox();self.fig_run=QComboBox();self.fig_model=QComboBox();self.fig_horizon=QSpinBox();self.fig_horizon.setRange(1,720)
        self.fig_width=QSpinBox();self.fig_width.setRange(60,400);self.fig_width.setValue(180);self.fig_height=QSpinBox();self.fig_height.setRange(40,400);self.fig_height.setValue(100)
        self.fig_dpi=combo([('300 dpi',300),('600 dpi',600),('1200 dpi',1200)]);self.fig_font=combo([('맑은 고딕','Malgun Gothic'),('Arial','Arial'),('Times New Roman','Times New Roman'),('DejaVu Sans','DejaVu Sans')])
        self.fig_size=QSpinBox();self.fig_size.setRange(6,24);self.fig_size.setValue(9)
        self.fig_title=QLineEdit();self.fig_title.setPlaceholderText('그림 제목 (생략 가능)');self.fig_xlabel=QLineEdit('Time');self.fig_ylabel=QLineEdit('Value')
        self.fig_start=QLineEdit();self.fig_start.setPlaceholderText('시작 날짜 (전체는 비움)');self.fig_end=QLineEdit();self.fig_end.setPlaceholderText('종료 날짜 (전체는 비움)')
        self.fig_gray=QCheckBox('흑백 인쇄');self.fig_legend=QCheckBox('범례 표시');self.fig_legend.setChecked(True)
        fields=[('그림 종류',self.fig_kind),('센서',self.fig_sensor),('학습 실행',self.fig_run),('단일 모델 · 산점도/잔차',self.fig_model),('예측 간격 · 시간',self.fig_horizon),('너비 · mm',self.fig_width),('높이 · mm',self.fig_height),('해상도',self.fig_dpi),('글꼴',self.fig_font),('글자 크기 · pt',self.fig_size),('시작',self.fig_start),('종료',self.fig_end),('제목',self.fig_title),('X축 제목',self.fig_xlabel),('Y축 제목 · 단위 포함',self.fig_ylabel)]
        for i,(t,w) in enumerate(fields):grid.addWidget(label(t,'muted'),(i//3)*2,i%3);grid.addWidget(w,(i//3)*2+1,i%3)
        grid.setHorizontalSpacing(18);cv.addLayout(grid);rr=QHBoxLayout();rr.addWidget(self.fig_gray);rr.addWidget(self.fig_legend);rr.addStretch();rr.addWidget(button('미리보기',self.preview_figure));rr.addWidget(button('Figure 내보내기',self.save_figure,True));cv.addLayout(rr);v.addWidget(c)
        self.fig_hint=label('관측 시계열은 센서별 시간 집계, 상관행렬은 그 값의 일평균(12시간 이상 관측)입니다. 실제 관측이 없는 구간은 채우지 않습니다.','muted');v.addWidget(self.fig_hint)
        self.plot_host=QWidget();self.plot_layout=QVBoxLayout(self.plot_host);self.plot_layout.setContentsMargins(0,0,0,0);self.plot_host.setMinimumHeight(340);self.plot_layout.addWidget(label('미리보기를 누르면 여기에 그림이 표시됩니다.','muted'));v.addWidget(self.plot_host,1)
        self.fig_run.currentIndexChanged.connect(self.refresh_figure_models)
        self.fig_kind.currentIndexChanged.connect(self.figure_defaults);self.fig_sensor.currentIndexChanged.connect(self.figure_defaults)

    def figure_defaults(self,*args):
        kind=self.fig_kind.currentData();sensor=self.fig_sensor.currentData();unit=''
        if self.project and self.project.latest_dataset():unit=self.project.latest_dataset()['columns'].get(sensor,{}).get('unit','')
        suffix=f' ({unit})' if unit else ''
        axes={'series':('Time',(sensor or 'Observed')+suffix),'comparison':('Time',(sensor or 'Value')+suffix),
              'scatter':('Observed'+suffix,'Predicted'+suffix),'residual':('Time','Residual'+suffix),'horizon':('Forecast horizon (h)','RMSE'+suffix),
              'correlation':('',''),'missing':('','')}
        x,y=axes[kind];self.fig_xlabel.setText(x);self.fig_ylabel.setText(y)
        result=kind in ('comparison','scatter','residual','horizon')
        self.fig_run.setEnabled(result);self.fig_model.setEnabled(kind in ('scatter','residual'));self.fig_horizon.setEnabled(kind in ('comparison','scatter','residual'))

    def navigate(self,n):
        self.stack.setCurrentIndex(n)
        for i,b in enumerate(self.nav):b.setChecked(i==n)

    def open_review(self):
        from .review_ui import ReviewWindow
        self.review_window=ReviewWindow();self.review_window.show();self.review_window.raise_();self.review_window.activateWindow()

    def open_manual(self):
        candidates=[Path(__file__).resolve().parents[1]/'README',Path(__file__).resolve().parents[2]/'README']
        path=next((p for p in candidates if p.exists()),None)
        if not path:return self.error('프로그램 폴더의 README를 확인하세요.')
        dialog=QDialog(self);dialog.setWindowTitle('Heritage Studio · 사용 설명서');dialog.resize(900,740)
        layout=QVBoxLayout(dialog);view=QTextBrowser();view.setOpenExternalLinks(True);view.setPlainText(path.read_text(encoding='utf-8'));layout.addWidget(view)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec()

    def busy(self):return self.process is not None and self.process.state()!=QProcess.ProcessState.NotRunning
    def require_project(self):
        if self.project:return True
        QMessageBox.information(self,'프로젝트 필요','먼저 새 프로젝트를 만들거나 백업을 복원하세요.');return False
    def idle(self):
        if self.busy():QMessageBox.information(self,'작업 진행 중','현재 작업이 끝나거나 취소된 후 실행하세요.');return False
        return True
    def error(self,e):QMessageBox.warning(self,'확인해 주세요',str(e))

    def refresh_projects(self,select=None):
        self.project_combo.blockSignals(True);self.project_combo.clear()
        for p in projects():self.project_combo.addItem(p.name,str(p.path))
        idx=self.project_combo.findData(str(select)) if select else 0
        self.project_combo.setCurrentIndex(max(0,idx));self.project_combo.blockSignals(False);self.select_project()
    def select_project(self,*args):
        path=self.project_combo.currentData();self.project=Project(path) if path else None
        if self.project:
            for r in self.project.list('run'):
                if r['status']=='running':r['status']='interrupted';r['error']='완료 전에 프로그램 또는 작업이 종료되었습니다.';self.project.update(r['id'],r)
        self.refresh()
    def new_project(self):
        if not self.idle():return
        name,ok=QInputDialog.getText(self,'새 프로젝트','문화유산 또는 연구 대상 이름')
        if ok and name.strip():p=Project.create(name);self.refresh_projects(p.path);self.navigate(1)
    def restore_project(self):
        if not self.idle():return
        file,_=QFileDialog.getOpenFileName(self,'프로젝트 백업 복원','','Heritage backup (*.heritage *.zip)')
        if file:
            try:p=Project.restore(file);self.refresh_projects(p.path);self.job_label.setText('백업 검증 및 복원 완료')
            except Exception as e:self.error(e)
    def export_project(self):
        if not self.require_project() or not self.idle():return
        f,_=QFileDialog.getSaveFileName(self,'프로젝트 내보내기',self.project.name+'.heritage','Heritage backup (*.heritage)')
        if f:self.start_job('backup',{'destination':f})
    def import_file(self):
        if not self.require_project() or not self.idle():return
        path,_=QFileDialog.getOpenFileName(self,'계측 데이터 선택','','Excel / CSV (*.xlsx *.csv)')
        if path:
            dialog=ImportDialog(path,self)
            if dialog.exec()==QDialog.DialogCode.Accepted:self.start_job('import',dialog.req)

    def refresh(self):
        ds=self.project.list('dataset') if self.project else [];models=self.project.list('model') if self.project else [];runs=self.project.list('run') if self.project else [];preds=self.project.list('prediction') if self.project else []
        latest=ds[0] if ds else None
        form_key=(str(self.project.path),latest['id']) if latest else None
        same_form=form_key is not None and self.form_dataset==form_key
        previous_targets=self.checked(self.targets);previous_envs=self.checked(self.envs)
        for w in (self.dataset_combo,self.run_combo,self.fig_run):w.blockSignals(True);w.clear()
        for d in ds:self.dataset_combo.addItem(f'{d["created"][:10]} · {d["source_name"]} · {d["rows"]:,}시각',d['id'])
        for r in runs:
            name=f'{r["created"][:16]} · {r["status"]} · {len(r["models"])} 모델';self.run_combo.addItem(name,r['id'])
            if r['status']=='complete':self.fig_run.addItem(name,r['id'])
        for w in (self.dataset_combo,self.run_combo,self.fig_run):w.blockSignals(False)
        self.targets.clear();self.envs.clear();self.fig_sensor.clear()
        if latest:
            if self.project.setting('demo'):self.minimum.setValue(1)
            for name,meta in latest['columns'].items():
                target=meta['role']=='target';widget=self.targets if target else self.envs;item=QListWidgetItem(f'{name}  [{meta["unit"]}]');item.setData(Qt.ItemDataRole.UserRole,name)
                selected=name in (previous_targets if target else previous_envs) if same_form else (target and self.targets.count()==0) or not target
                item.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked);widget.addItem(item)
                self.fig_sensor.addItem(name,name)
            start=pd.Timestamp(latest['start']);end=pd.Timestamp(latest['end']);span=end-start
            if not same_form:self.train_end.setText(str((start+span*.7).floor('h'))[:19]);self.val_end.setText(str((start+span*.85).floor('h'))[:19])
            self.home_detail.setText(f'{self.project.name}\n최근 관측: {latest["end"]}  ·  {latest["rows"]:,}개 시각\n새 모델 학습 전 센서별 결측과 단위를 확인하세요. 생성일과 마지막 관측일은 다를 수 있습니다.')
        else:self.home_detail.setText('아직 데이터가 없습니다. 엑셀·CSV를 등록하거나 연습용 예제를 만들어 보세요.')
        self.form_dataset=form_key
        vals=[len(ds),sum(m['role']=='target' for m in latest['columns'].values()) if latest else 0,len([m for m in models if m.get('status')=='complete' and any(r['id']==m['run_id'] and r['status']=='complete' for r in runs)]),len(preds)]
        for w,n in zip(self.home_values,vals):w.setText(f'{n:,}')
        self.model_rows=models;table=[]
        for m in models:
            vm=[x['MAE'] for x in m['metrics'] if x['split']=='validation'];default=self.project.setting('default:'+m['sensor'])==m['id']
            table.append({'기본':'●' if default else '', '센서':m['sensor'],'모델':m['model'],'입력':m['mode'],'검증 평균 MAE':np.mean(vm),'학습 종료':m['train_end'],'상태':m['status']})
        set_table(self.models_table,pd.DataFrame(table));self.models_table.selectionModel().selectionChanged.connect(self.show_model_detail);self.pred_rows=preds
        set_table(self.predictions_table,pd.DataFrame([{'센서':p['sensor'],'예측 기준':p['origin'],'마지막 목표':p['last_target'],'생성일':p['created'][:19]} for p in preds]))
        self.show_quality();self.show_metrics();self.refresh_figure_models();self.evaluation_label.setText('예측 생성 후 실제값이 추가되면 사후 평가를 실행할 수 있습니다.')
        if self.project:
            evaluations=self.project.list('evaluation')
            if evaluations:self.evaluation_label.setText('최근 사후 평가: '+json.dumps(evaluations[0]['metrics'],ensure_ascii=False))

    def show_quality(self,*args):
        if not self.project or not self.dataset_combo.currentData():set_table(self.quality_table,pd.DataFrame());return
        d=self.project.get(self.dataset_combo.currentData());q=pd.DataFrame(d['quality']).rename(columns={'sensor':'센서','count':'유효 관측','missing_pct':'결측 %','first':'첫 관측','last':'마지막 관측','min':'최솟값','max':'최댓값'})
        set_table(self.quality_table,q);self.data_summary.setText(f'{d["start"]} → {d["end"]}  |  {d["rows"]:,}시각  |  충돌 {d["conflicts"]:,}셀  |  해석 불가 날짜 {d["invalid_dates"]:,}행')
    def checked(self,w):return [w.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.count()) if w.item(i).checkState()==Qt.CheckState.Checked]
    def start_training(self):
        if not self.require_project() or not self.idle():return
        d=self.project.latest_dataset()
        if not d:return self.error('먼저 데이터를 등록하세요.')
        try:
            from .engine import validate_config
            cfg={'dataset_id':d['id'],'targets':self.checked(self.targets),'environment':self.checked(self.envs),'modes':(['self'] if self.self_mode.isChecked() else [])+(['environment'] if self.env_mode.isChecked() else []),
                 'models':['persistence']+[k for k,w in self.model_checks.items() if w.isChecked()], 'lookback':self.lookback.value(),'horizons':[int(x.strip()) for x in self.horizons.text().split(',')],
                 'epochs':self.epochs.value(),'minimum':self.minimum.value(),'seed':self.seed.value(),'device':self.device.currentData(), 'train_end':str(pd.Timestamp(self.train_end.text())),'val_end':str(pd.Timestamp(self.val_end.text()))}
            validate_config(cfg)
            n=len(cfg['targets'])*len(cfg['modes'])*len(cfg['models'])
            if QMessageBox.question(self,'학습 조건 확인',f'최신 데이터 버전에서 {len(cfg["targets"])}개 센서, {n}개 모델 조건을 실행합니다.\n기존 모델과 결과는 보존됩니다. 진행할까요?')!=QMessageBox.StandardButton.Yes:return
            self.start_job('train',cfg)
        except Exception as e:self.error(e)
    def show_metrics(self,*args):
        if not self.project or not self.run_combo.currentData():set_table(self.metrics_table,pd.DataFrame());return
        r=self.project.get(self.run_combo.currentData());p=self.project.file(r['folder']+'/metrics.csv')
        if p.exists():
            df=pd.read_csv(p);set_table(self.metrics_table,df[['sensor','model','mode','horizon','split','n','MAE','RMSE','R2','skill_mae']])
        else:set_table(self.metrics_table,pd.DataFrame([{'상태':r['status'],'안내':r.get('error','학습 결과 준비 중')}]))
    def selected(self,view,rows):
        ix=view.currentIndex().row()
        if ix<0 or ix>=len(rows):raise ValueError('표에서 항목을 먼저 선택하세요.')
        return rows[ix]
    def show_model_detail(self,*args):
        try:
            m=self.selected(self.models_table,self.model_rows);run=self.project.get(m['run_id'])
            self.model_hint.setText(f'{m["sensor"]} · {m["model"]} / {m["mode"]} · 입력 {m["lookback"]}시간 · 예측 {m["horizons"]}시간\n입력 센서: {", ".join(m["inputs"])} · 데이터 {m["dataset_id"][:8]} · 실행 {run["status"]}\n검증 종료: {m["val_end"]} · 단위: {m["columns_meta"][m["sensor"]]["unit"]}')
        except ValueError:pass
    def assign_model(self):
        if not self.require_project() or not self.idle():return
        try:
            m=self.selected(self.models_table,self.model_rows)
            if m['status']!='complete' or self.project.get(m['run_id'])['status']!='complete':raise ValueError('완료된 학습의 모델만 지정할 수 있습니다.')
            old=self.project.setting('default:'+m['sensor']);self.project.add('assignment',{'sensor':m['sensor'],'model_id':m['id'],'previous':old});self.project.set_setting('default:'+m['sensor'],m['id']);self.refresh();self.job_label.setText(m['sensor']+'의 기본 모델을 지정했습니다.')
        except Exception as e:self.error(e)
    def predict_selected(self):
        if not self.require_project() or not self.idle():return
        try:m=self.selected(self.models_table,self.model_rows);self.start_job('predict',{'model_id':m['id'],'origin':self.pred_origin.text().strip() or None})
        except Exception as e:self.error(e)
    def evaluate_selected(self):
        if not self.require_project() or not self.idle():return
        try:p=self.selected(self.predictions_table,self.pred_rows);self.start_job('evaluate',{'prediction_id':p['id']})
        except Exception as e:self.error(e)

    def export_quality(self):
        if not self.require_project() or not self.dataset_combo.currentData():return
        d=self.project.get(self.dataset_combo.currentData());self.save_csv(pd.DataFrame(d['quality']),'data_quality.csv')
    def export_metrics(self):
        if not self.require_project() or not self.run_combo.currentData():return
        r=self.project.get(self.run_combo.currentData());path=self.project.file(r['folder']+'/metrics.csv')
        if path.exists():self.save_csv(pd.read_csv(path),'model_metrics.csv')
    def export_prediction(self):
        if not self.require_project():return
        try:p=self.selected(self.predictions_table,self.pred_rows);self.save_csv(pd.read_csv(self.project.file(p['file'])),'forecast.csv')
        except Exception as e:self.error(e)
    def save_csv(self,df,name):
        f,_=QFileDialog.getSaveFileName(self,'표 내보내기',name,'CSV (*.csv);;Excel (*.xlsx)')
        if f:
            try:
                if f.lower().endswith('.xlsx'):df.to_excel(f,index=False)
                else:df.to_csv(f,index=False,encoding='utf-8-sig')
                self.job_label.setText('저장 완료: '+f)
            except Exception as e:self.error(e)

    def refresh_figure_models(self,*args):
        self.fig_model.clear();run=self.fig_run.currentData()
        if not self.project:return
        for m in self.project.list('model'):
            if m['run_id']==run:self.fig_model.addItem(f'{m["sensor"]} / {m["model"]} / {m["mode"]}',m['id'])
    def figure_inputs(self):
        from .figures import FigureSpec
        from .data import hourly
        if not self.project:raise ValueError('프로젝트를 먼저 선택하세요.')
        kind=self.fig_kind.currentData();sensor=self.fig_sensor.currentData();meta={'project':self.project.name,'sensor':sensor,'kind':kind};raw,version=self.project.load_data();meta['dataset_id']=version['id']
        if kind in ('series','correlation','missing'):
            frame=hourly(raw,minimum=self.minimum.value(),metadata=version['columns']);meta['minimum_per_hour']=self.minimum.value();meta['columns_meta']=version['columns']
            if kind=='series':data=frame[[sensor]]
            elif kind=='correlation':data=frame.resample('D').mean().where(frame.resample('D').count()>=12);meta['aggregation']='daily; >=12 valid hourly bins'
            else:data=frame
        else:
            runid=self.fig_run.currentData()
            if not runid:raise ValueError('먼저 학습을 완료하거나 관측 기반 그림을 선택하세요.')
            run=self.project.get(runid);meta['run_id']=runid;meta['dataset_id']=run['config']['dataset_id'];meta['split']='test'
            if kind=='horizon':
                data=pd.read_csv(self.project.file(run['folder']+'/metrics.csv'));data=data[(data.sensor==sensor)&(data.split=='test')]
            else:
                chunks=[]
                for chunk in pd.read_csv(self.project.file(run['folder']+'/predictions.csv.gz'),chunksize=200000):
                    selected=chunk[(chunk.sensor==sensor)&(chunk.split=='test')&(chunk.horizon==self.fig_horizon.value())]
                    if kind in ('scatter','residual'):selected=selected[selected.model_id==self.fig_model.currentData()]
                    if not selected.empty:chunks.append(selected)
                data=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
                meta['horizon']=self.fig_horizon.value();meta['model_id']=self.fig_model.currentData() if kind in ('scatter','residual') else None
        if data.empty:raise ValueError('조건에 맞는 자료가 없습니다. 센서·모델·예측간격을 확인하세요.')
        start=self.fig_start.text().strip();end=self.fig_end.text().strip()
        if start and end and pd.Timestamp(start)>pd.Timestamp(end):raise ValueError('시작은 종료 이전이어야 합니다.')
        if start or end:
            if kind=='horizon':raise ValueError('horizon 지표는 전체 테스트 기간입니다. 날짜 필터를 비워 주세요.')
            ix=pd.DatetimeIndex(pd.to_datetime(data.time)) if 'time' in data else data.index
            mask=np.ones(len(data),bool)
            if start:mask&=ix>=pd.Timestamp(start)
            if end:mask&=ix<=pd.Timestamp(end)
            data=data.loc[mask]
        if data.empty:raise ValueError('조건에 맞는 자료가 없습니다. 센서·모델·예측간격·기간을 확인하세요.')
        spec=FigureSpec(kind=kind,width_mm=self.fig_width.value(),height_mm=self.fig_height.value(),dpi=self.fig_dpi.currentData(),font_size=self.fig_size.value(),font_family=self.fig_font.currentData(),
                        title=self.fig_title.text(),xlabel=self.fig_xlabel.text(),ylabel=self.fig_ylabel.text(),grayscale=self.fig_gray.isChecked(),legend=self.fig_legend.isChecked())
        meta['start_filter']=start;meta['end_filter']=end
        return data,spec,meta
    def preview_figure(self):
        if not self.idle():return
        try:
            from .figures import make_figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            data,spec,meta=self.figure_inputs();fig=make_figure(data,spec)
            while self.plot_layout.count():
                item=self.plot_layout.takeAt(0)
                if item.widget():item.widget().deleteLater()
            self.figure_canvas=FigureCanvasQTAgg(fig);self.figure_canvas.setMinimumHeight(340);self.plot_layout.addWidget(self.figure_canvas);self.figure_canvas.draw()
            self.fig_hint.setText(f'{len(data):,}행 · {spec.width_mm} × {spec.height_mm} mm · {spec.dpi} dpi · 평활화/결측 보간 없음')
        except Exception as e:self.error(e)
    def save_figure(self):
        if not self.idle():return
        try:
            from .figures import export_figure
            data,spec,meta=self.figure_inputs()
            f,_=QFileDialog.getSaveFileName(self,'논문 Figure 저장','Figure_1.pdf','PDF vector (*.pdf);;SVG vector (*.svg);;PNG (*.png);;TIFF (*.tiff)')
            if f:export_figure(data,spec,f,meta);self.job_label.setText('Figure, 사용 데이터, 설정 JSON을 함께 저장했습니다.');self.preview_figure()
        except Exception as e:self.error(e)

    def start_job(self,kind,config):
        if not self.idle():return
        self.job_path=self.project.file('jobs/'+uid()+'.json');write_json(self.job_path,{'kind':kind,'project':str(self.project.path),'config':config})
        self.process=QProcess(self);self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if getattr(sys,'frozen',False):exe=sys.executable;args=['--worker',str(self.job_path)]
        else:exe=sys.executable;args=['-u',str(Path(__file__).resolve().parents[1]/'main.py'),'--worker',str(self.job_path)]
        self.buffer='';self.event_offset=0;self.event_timer.start();self.process.readyReadStandardOutput.connect(self.read_progress);self.process.finished.connect(self.job_finished);self.process.errorOccurred.connect(self.process_error)
        self.project_combo.setEnabled(False);self.progress.setRange(0,0);self.cancel_button.setEnabled(True);self.job_label.setText('작업 준비 중…');self.process.start(exe,args)
    def read_progress(self):
        if not self.process:return
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:
                json.loads(line)
                continue  # Progress is read from the journal, including in windowed EXE builds.
            except json.JSONDecodeError:message=line
            self.job_label.setText(message[-180:]);self.log.appendPlainText(message)
    def read_events(self):
        if not self.job_path:return
        path=self.job_path.with_suffix('.events.jsonl')
        if not path.exists():return
        with path.open('rb') as f:
            f.seek(self.event_offset)
            while True:
                line=f.readline()
                if not line or not line.endswith(b'\n'):break
                self.event_offset=f.tell()
                try:message=json.loads(line)['message']
                except (ValueError,KeyError):continue
                self.job_label.setText(message[-180:]);self.log.appendPlainText(message)
    def process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:self.job_finished(-1,QProcess.ExitStatus.CrashExit)
    def job_finished(self,*args):
        self.read_progress();self.read_events();self.event_timer.stop();self.progress.setRange(0,1);self.progress.setValue(1);self.cancel_button.setEnabled(False);self.project_combo.setEnabled(True)
        resultpath=self.job_path.with_suffix('.result.json')
        result=read_json(resultpath) if resultpath.exists() else {'ok':False,'error':'작업 프로세스가 중단되었습니다. jobs 폴더의 로그를 확인하세요.'}
        self.process=None
        for r in self.project.list('run'):
            if r['status']=='running':r['status']='interrupted';r['error']=result.get('error','작업 중단');self.project.update(r['id'],r)
        self.refresh()
        if result['ok']:self.job_label.setText('완료 · 결과가 프로젝트에 저장되었습니다.')
        elif result.get('cancelled'):self.job_label.setText('취소됨 · 기존 데이터와 모델은 유지됩니다.')
        else:self.job_label.setText('작업을 완료하지 못했습니다.');self.error(result['error'])
    def cancel_job(self):
        if self.busy():
            self.job_path.with_suffix('.cancel').touch();self.cancel_button.setEnabled(False);self.job_label.setText('취소 요청 중 · 현재 계산을 정리합니다…')
            p=self.process;QTimer.singleShot(15000,lambda:p.kill() if p.state()!=QProcess.ProcessState.NotRunning else None)
    def closeEvent(self,event):
        if self.busy():
            QMessageBox.information(self,'작업 진행 중','작업을 취소하거나 완료한 후 프로그램을 닫아 주세요.');event.ignore()
        else:event.accept()
    def demo_project(self):
        if not self.idle():return
        try:
            from .demo import create_demo
            p=create_demo();self.refresh_projects(p.path);self.minimum.setValue(1);self.job_label.setText('합성 연습용 자료입니다. 실제 연구 결과로 사용하지 마세요.');self.navigate(1)
        except Exception as e:self.error(e)
