"""Exercise the user's complete flow against an isolated synthetic project."""
import os,time,json
from pathlib import Path
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from heritage.demo import create_demo
from heritage.guided import MainWindow


def run(destination):
    base=Path(destination).resolve();base.mkdir(parents=True,exist_ok=True);os.environ['HERITAGE_HOME']=str(base)
    project=create_demo(base/'projects');app=QApplication([]);w=MainWindow();errors=[];w.error=lambda e:errors.append(str(e));w.show()
    def settle():
        end=time.time()+120
        while (w.busy() or w.readers) and time.time()<end:app.processEvents();time.sleep(.02)
        for _ in range(20):app.processEvents();time.sleep(.02)
        assert not w.busy() and not w.readers and not errors,errors
    settle();assert len(w.measurements)==1440
    w.browse_all();assert w.raw_table.model().rowCount()==1440
    w.forecast_duration.setValue(3);w.lookback.setValue(24);w.epochs.setValue(1);w.model_width.setCurrentIndex(0);w.layer_count.setValue(2);w.learning_rate.setValue(.002)
    w.model_checks['lstm'].setChecked(True);w.start_training();settle();assert w.checked_config is not None
    assert not project.list('run')
    w.start_training();settle();assert w.stack.currentIndex()==5
    assert w.metrics_table.model().rowCount()==2 and w.compare_layout.count()>=2
    w.results_to_future();w.future_duration.setValue(2);w.make_future();settle()
    assert len(project.list('prediction'))==1
    rec=project.list('prediction')[0];assert rec['requested_hours']==48
    assert w.predict_tabs.currentWidget()==w.future_plot_tab and w.future_figure.axes[0].lines[1].get_xdata().size==3
    QFileDialog.getSaveFileName=lambda *args,**kwargs:(str(base/'future.svg'),'SVG (*.svg)')
    w.save_future_plot();assert (base/'future.svg').exists() and (base/'future.csv').exists()
    for width,height in [(1440,920),(1100,760)]:
        w.resize(width,height)
        for n,name in [(1,'data'),(2,'training'),(5,'results'),(3,'future')]:
            w.navigate(n);settle();w.grab().save(str(base/f'{name}-{width}.png'))
        for tab in range(5):
            w.navigate(2);w.training_steps.setCurrentIndex(tab);settle();w.grab().save(str(base/f'step{tab+1}-{width}.png'))
    QMessageBox.question=lambda *args,**kwargs:QMessageBox.StandardButton.Yes
    w.import_table.selectRow(0);w.change_import(False);settle();assert project.latest_dataset() is None and w.measurements is None
    w.import_table.selectRow(0);w.change_import(True);settle();assert len(w.measurements)==1440
    w.close();print(json.dumps({'ok':True,'models':len(project.list('model')),'predictions':len(project.list('prediction')),'errors':errors}))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
