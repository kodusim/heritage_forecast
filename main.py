import os
import sys
from pathlib import Path


def report_exception(kind,value,tb):
    import traceback
    detail=''.join(traceback.format_exception(kind,value,tb))
    try:
        from heritage.storage import root_dir
        root_dir().mkdir(parents=True,exist_ok=True)
        (root_dir()/'application_error.log').write_text(detail,encoding='utf-8')
    except Exception:pass
    if sys.stderr is not None:sys.stderr.write(detail)
    if '--worker' not in sys.argv:
        try:
            from PySide6.QtWidgets import QApplication,QMessageBox
            if QApplication.instance():QMessageBox.warning(None,'오류 안내',str(value)+'\n\n작업 폴더의 application_error.log에 원인을 기록했습니다.')
        except Exception:pass

def main():
    if '--worker' in sys.argv:
        from heritage.worker import main as worker
        return worker(sys.argv[sys.argv.index('--worker')+1])
    from PySide6.QtCore import QLockFile
    from PySide6.QtWidgets import QApplication,QMessageBox
    from heritage.storage import root_dir
    app=QApplication(sys.argv)
    app.setApplicationName('Heritage Studio');app.setOrganizationName('Heritage Research')
    root_dir().mkdir(parents=True,exist_ok=True)
    lock=QLockFile(str(root_dir()/('results.lock' if '--results' in sys.argv else 'application.lock')));lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None,'이미 실행 중','Heritage Studio가 이미 실행 중입니다. 열려 있는 창을 확인하세요.');return 0
    from heritage.review import review_projects
    if '--results' in sys.argv:
        from heritage.review_ui import ReviewWindow
        window=ReviewWindow()
    else:
        from heritage.studio import MainWindow
        window=MainWindow()
    window.show()
    if '--screenshot' in sys.argv:
        from PySide6.QtCore import QTimer
        output=sys.argv[sys.argv.index('--screenshot')+1]
        def capture():window.grab().save(output);app.quit()
        QTimer.singleShot(1200,capture)
    return app.exec()

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    sys.excepthook=report_exception
    try:exit_code=main()
    except Exception:
        report_exception(*sys.exc_info());exit_code=1
    raise SystemExit(exit_code)
