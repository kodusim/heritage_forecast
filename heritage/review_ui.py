from pathlib import Path
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QFrame,QComboBox,
    QPushButton,QScrollArea,QFileDialog,QMessageBox,QDialog,QTextBrowser,QDialogButtonBox,QApplication)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from .review import review_projects,load_review,make_panels,panel_figure,overview_figure,metric_label,export_overview,COLORS
from .ui import STYLE,label,button,combo

class ReviewWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Transformer 결과 · 실제값과 예측값 비교');self.resize(1500,980);self.setMinimumSize(1060,720);self.setStyleSheet(STYLE)
        self.reviews=[];self.panels=[];self.management=None
        host=QWidget();self.setCentralWidget(host);layout=QVBoxLayout(host);layout.setContentsMargins(26,22,26,20);layout.setSpacing(12)
        top=QHBoxLayout();heading=QVBoxLayout();heading.addWidget(label('Transformer 결과, 센서별로 바로 비교','pageTitle'))
        heading.addWidget(label('검정은 실제값, 색상 선은 Transformer 예측값입니다. 센서를 누르면 상세 그래프가 열립니다.','muted'));top.addLayout(heading,1)
        top.addWidget(button('데이터·학습 관리',self.open_management));top.addWidget(button('비교 조건 보기',self.show_conditions));layout.addLayout(top)
        controls=QFrame();controls.setObjectName('card');row=QHBoxLayout(controls);row.setContentsMargins(16,12,16,12)
        self.group=QComboBox();self.group.addItem('전체 고분',None)
        self.horizon=combo([('24시간 뒤',24),('6시간 뒤',6),('1시간 뒤',1)])
        self.mode=combo([('거동 + 내부 온도','environment'),('거동 과거값만','self'),('두 입력 조건 함께','both')])
        self.period=combo([('테스트 전체',0),('마지막 30일 확대',30),('마지막 7일 확대',7)])
        self.view=combo([('전체 센서 한눈에','overview'),('큰 카드로 보기','cards')])
        for text,widget in [('대상',self.group),('예측',self.horizon),('입력',self.mode),('표시 기간',self.period)]:row.addWidget(label(text,'muted'));row.addWidget(widget)
        row.addWidget(self.view)
        row.addStretch();row.addWidget(button('새로고침',self.reload));row.addWidget(button('비교 그림 저장',self.save,True));layout.addWidget(controls)
        self.summary=label('저장된 실제 데이터와 Transformer 결과를 읽고 있습니다.','muted');layout.addWidget(self.summary)
        legend=QHBoxLayout();self.legend_items={}
        for key,text,color in [('actual','● 실제값',COLORS['actual']),('environment','● Transformer · 거동 + 내부 온도',COLORS['environment']),('self','┄ Transformer · 거동 과거값만',COLORS['self'])]:
            item=label(text,wrap=False);item.setStyleSheet(f'color:{color};font-weight:600;');legend.addWidget(item);self.legend_items[key]=item
        legend.addStretch();legend.addWidget(label('센서를 누르면 크게 볼 수 있습니다.','muted',wrap=False));layout.addLayout(legend)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True);layout.addWidget(self.scroll,1)
        self.footer=label('MAE 개선이 양수면 마지막 값 유지 기준선보다 오차가 작습니다. 음수면 기준선보다 나쁩니다.','muted');layout.addWidget(self.footer)
        for c in (self.group,self.horizon,self.mode,self.period,self.view):c.currentIndexChanged.connect(self.render)
        QTimer.singleShot(50,self.reload)

    def reload(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.reviews=[load_review(p) for p in review_projects()]
            current=self.group.currentData();self.group.blockSignals(True);self.group.clear();self.group.addItem('2·3번 고분 전체' if any(r.project.setting('customer_review') for r in self.reviews) else '전체 프로젝트',None)
            for review in self.reviews:self.group.addItem(review.group,review.project.path.name)
            index=self.group.findData(current);self.group.setCurrentIndex(max(0,index));self.group.blockSignals(False);self.render()
        except Exception as error:QMessageBox.warning(self,'결과 확인',str(error))
        finally:QApplication.restoreOverrideCursor()

    def render(self,*args):
        if not hasattr(self,'scroll'):return
        prepared=[r for r in self.reviews if r.project.setting('customer_review')]
        selected=[r for r in self.reviews if r.project.path.name==self.group.currentData()] if self.group.currentData() else (prepared or self.reviews)
        self.panels=make_panels(selected,self.horizon.currentData(),self.mode.currentData(),self.period.currentData())
        for condition in ('environment','self'):self.legend_items[condition].setVisible(self.mode.currentData() in (condition,'both'))
        body=QWidget();grid=QGridLayout(body);grid.setContentsMargins(0,0,5,0);grid.setSpacing(14)
        columns=3 if len(self.panels)>8 else 2
        valid=sum(bool(p.lines) for p in self.panels);improved=sum(((p.scores.get('environment') or p.scores.get('self') or {}).get('improvement_pct') or 0)>0 for p in self.panels)
        if self.panels:
            start=min(p.start for p in self.panels).strftime('%Y.%m.%d');end=max(p.end for p in self.panels).strftime('%Y.%m.%d')
            scope=' (지표: 온도 포함)' if self.mode.currentData()=='both' else ''
            self.summary.setText(f'센서 {len(self.panels)}개 · 비교 가능 {valid}개 · 기준선보다 MAE가 낮은 센서 {improved}개{scope}  |  {start}–{end}  |  매 시각 관측으로 {self.horizon.currentData()}시간 뒤 예측')
        else:
            self.summary.setText('아직 완료된 Transformer 결과가 없습니다. 데이터·학습 관리에서 자료를 등록하고 학습하세요.')
            grid.addWidget(label('학습이 완료되면 실제값과 예측값 그래프가 여기에 나타납니다.','muted'),0,0)
        if self.panels and self.view.currentData()=='overview':
            canvas=FigureCanvasQTAgg(overview_figure(self.panels,self.horizon.currentData(),screen=True));canvas.setMinimumSize(850,500)
            canvas.mpl_connect('button_press_event',lambda event:self.detail(event.inaxes._review_panel) if event.inaxes is not None and hasattr(event.inaxes,'_review_panel') else None)
            grid.addWidget(canvas,0,0)
            old=self.scroll.takeWidget()
            if old:old.deleteLater()
            self.scroll.setWidget(body);return
        for index,panel in enumerate(self.panels):
            card=QFrame();card.setObjectName('card');card.setMinimumHeight(262);v=QVBoxLayout(card);v.setContentsMargins(12,10,12,10);v.setSpacing(5)
            title=button(f'{panel.review.group} · {panel.sensor}  [{panel.unit}]    ↗',lambda checked=False,p=panel:self.detail(p));title.setStyleSheet('text-align:left;border:0;background:transparent;padding:2px;font-size:14px;font-weight:700;');v.addWidget(title)
            figure=FigureCanvasQTAgg(panel_figure(panel));figure.setMinimumHeight(180);figure.setMaximumHeight(205);v.addWidget(figure,1)
            score=panel.scores.get('environment') or panel.scores.get('self')
            text=metric_label(score,panel.unit) if score else panel.reason.replace('\n',' · ')
            score_label=label(text);score_label.setStyleSheet('font-size:11px;color:'+('#a15a18' if score and score.get('improvement_pct') is not None and score['improvement_pct']<0 else '#526579')+';');v.addWidget(score_label)
            grid.addWidget(card,index//columns,index%columns)
        for c in range(columns):grid.setColumnStretch(c,1)
        old=self.scroll.takeWidget()
        if old:old.deleteLater()
        self.scroll.setWidget(body)

    def detail(self,panel):
        dialog=QDialog(self);dialog.resize(1180,760);dialog.setWindowTitle(f'{panel.review.group} · {panel.sensor} · 실제값과 예측값')
        v=QVBoxLayout(dialog);v.addWidget(label(f'{panel.review.group} · {panel.sensor} · {self.horizon.currentData()}시간 앞 예측','pageTitle'))
        canvas=FigureCanvasQTAgg(panel_figure(panel));canvas.figure.axes[0].legend(frameon=False,loc='best',prop={'family':['Malgun Gothic','DejaVu Sans'],'size':9})
        v.addWidget(NavigationToolbar2QT(canvas,dialog));v.addWidget(canvas,1)
        for condition,score in panel.scores.items():v.addWidget(label(('거동 + 내부 온도' if condition=='environment' else '거동 과거값만')+' : '+metric_label(score,panel.unit)))
        if panel.reason:v.addWidget(label(panel.reason,'muted'))
        close=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);close.rejected.connect(dialog.reject);v.addWidget(close);dialog.exec()

    def show_conditions(self):
        text=[]
        for review in self.reviews:
            if not review.run:continue
            cfg=review.run['config'];text.append(f'{review.group}\n학습 종료: {cfg["train_end"]}\n검증 종료: {cfg["val_end"]}\n입력: 과거 {cfg["lookback"]}시간 · 최대 {cfg["epochs"]} epoch, 검증 손실 조기 종료\n환경 입력: {", ".join(cfg["environment"])}\n데이터 버전: {review.version["id"]}\n실행: {review.run["id"]}')
        QMessageBox.information(self,'현재 그래프의 비교 조건','\n\n'.join(text)+'\n\n검증 이후 테스트 실제값으로 비교합니다. 결측값은 보간하지 않습니다.\n두 입력 조건과 기준선은 같은 평가 시각을 사용합니다. 여러 센서·조건·seed를 반복 검증한 최종 성능 확정 결과는 아닙니다.')

    def save(self):
        if not self.panels:return
        path,_=QFileDialog.getSaveFileName(self,'센서별 비교 그림 저장',f'Transformer_{self.horizon.currentData()}h.png','PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_overview(self.panels,self.horizon.currentData(),path);self.footer.setText('저장 완료 · 그림, 실제값·예측값 CSV, 지표표, 비교 조건 JSON을 함께 저장했습니다.')
        except Exception as error:QMessageBox.warning(self,'그림 저장',str(error))
        finally:QApplication.restoreOverrideCursor()

    def open_management(self):
        from .ui import MainWindow
        if self.management is None:self.management=MainWindow()
        self.management.show();self.management.raise_();self.management.activateWindow()
