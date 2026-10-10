"""Real Qt interactions against a disposable project; no changes to user projects."""
import os,time,json
from pathlib import Path
from PySide6.QtWidgets import QApplication,QMessageBox,QFileDialog
from heritage.demo import create_demo
from heritage.engine import train
from heritage.studio import MainWindow

def run(base):
    base=Path(base).resolve();base.mkdir(parents=True,exist_ok=True);os.environ['HERITAGE_HOME']=str(base)
    project=create_demo(base/'projects');df,d=project.load_data()
    train(project,{'dataset_id':d['id'],'targets':['DEMO-D1'],'environment':['Temperature'],'modes':['self','environment'],
        'models':['persistence','linear','lstm','transformer'],'lookback':12,'horizons':[1,6,24],
        'train_end':str(df.index[900]),'val_end':str(df.index[1150]),'minimum':1,'epochs':1,'seed':42,'device':'cpu'})
    app=QApplication.instance() or QApplication([]);w=MainWindow();errors=[];w.error=lambda e:errors.append(str(e));w.show();app.processEvents()
    def wait_job():
        limit=time.time()+90
        while w.busy() and time.time()<limit:app.processEvents();time.sleep(.03)
        for _ in range(5):app.processEvents();time.sleep(.03)
        assert not w.busy(),'worker timeout'
        assert not errors,errors
    QMessageBox.question=lambda *args,**kwargs:QMessageBox.StandardButton.Yes
    w.epochs.setValue(1);w.env_mode.setChecked(True);w.start_training();wait_job()
    assert w.stack.currentIndex()==5
    w.navigate(5);assert len(w.metrics_table.model().df)==1
    w.comparison_plot();assert w.compare_layout.count()==2
    w.navigate(3)
    ix=next(i for i,m in enumerate(w.model_rows) if m['model']=='transformer')
    w.models_table.selectRow(ix);w.check_origins();wait_job()
    assert w.predict_button.isEnabled() and len(w.valid_times)>0
    w.predict_selected();wait_job();assert len(w.pred_rows)==1
    w.evaluate_selected();wait_job();assert project.list('evaluation')
    w.navigate(4);w.preview_figure();assert w.figure_canvas is not None
    QFileDialog.getSaveFileName=lambda *args,**kwargs:(str(base/'paper.svg'),'SVG vector (*.svg)')
    w.save_figure();assert (base/'paper.svg').exists() and (base/'paper_figure.json').exists()
    assert not errors,errors
    for width,height in [(1440,920),(1100,760)]:
        w.resize(width,height)
        for n,name in [(1,'data'),(2,'train'),(5,'results'),(3,'predict'),(4,'figures')]:
            w.navigate(n)
            if n==5:w.result_tabs.setCurrentIndex(0)
            if n==3:w.predict_tabs.setCurrentIndex(0)
            app.processEvents();w.grab().save(str(base/f'{name}-{width}.png'))
        for index in (1,2):
            w.navigate(2);w.training_steps.setCurrentIndex(index);app.processEvents();w.grab().save(str(base/f'train-step{index+1}-{width}.png'))
    w.import_table.selectRow(0);w.change_import(False);wait_job();assert project.latest_dataset() is None
    w.import_table.selectRow(0);w.change_import(True);wait_job();assert project.latest_dataset()
    w.close();print(json.dumps({'ok':True,'models':len(project.list('model')),'predictions':len(project.list('prediction')),'errors':errors}))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
