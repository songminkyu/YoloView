import time

from utils import glo
import json
import os
from ui.YOLOSHOWUI import Ui_MainWindow
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow
from yoloshow.YOLOThreadPool import YOLOThreadPool
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE, MODEL_THREAD_CLASSES

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUI.ui"
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting', 'src_distribute']


# # YOLOSHOW 윈도우 클래스는 UI 파일과 Ui_mainWindow를 동적으로 로드합니다.
class YOLOSHOW(QMainWindow, YOLOSHOWBASE):
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.current_workpath = os.getcwd()
        self.inputPath = None
        self.result_statistic = None
        self.detect_result = None
        self.detect_errors = None
        self.view_mode = "single"

        # --- UI 로드 --- #
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 투명한 배경
        self.setWindowFlags(Qt.FramelessWindowHint)  # 헤드리스 창
        self.initSiderWidget()
        # --- UI 로드 --- #

        # --- 최대화 최소화 닫기 --- #
        self.ui.maximizeButton.clicked.connect(self.maxorRestore)
        self.ui.minimizeButton.clicked.connect(self.showMinimized)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.topbox.doubleClickFrame.connect(self.maxorRestore)
        # --- 최대화 최소화 닫기 --- #

        # --- 재생 일시 정지 --- #
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/play.png"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Active,
                                QtGui.QIcon.On)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"),
                                QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.ui.run_button.setCheckable(True)
        self.ui.run_button.setIcon(self.playIcon)
        # --- 재생 일시 정지 --- #

        # --- track mode --- #
        self.ui.track_box.currentIndexChanged.connect(self.selectedTrackMode)
        # --- track mode --- #

        # --- 사이드바 확대/축소 --- #
        self.ui.src_menu.clicked.connect(self.scaleMenu)  # hide menu button
        self.ui.src_setting.clicked.connect(self.scalSetting)  # setting button
        # --- 사이드바 확대/축소 --- #

        # --- PT 모델 자동 로드/동적으로 변경 --- #
        self.pt_Path = f"{self.current_workpath}/ptfiles/"
        os.makedirs(self.pt_Path, exist_ok=True)
        self.pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
        self.ui.model_box.clear()
        self.ui.model_box.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.ui.model_box.currentTextChanged.connect(self.changeModel)
        # --- PT 모델 자동 로드/동적으로 변경 --- #

        # --- 사진/동영상 가져오기, 카메라 호출, 폴더 가져오기(일괄 처리), 웹 카메라 호출, 결과 통계 사진, 결과 통계 테이블 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        self.ui.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        self.ui.src_result.clicked.connect(self.showResultStatics)
        self.ui.src_table.clicked.connect(self.showTableResult)
        self.ui.src_curation.clicked.connect(self.datasetCuration)
        # --- 사진/동영상 가져오기, 카메라 호출, 폴더 가져오기(일괄 처리), 웹 카메라 호출, 결과 통계 사진, 결과 통계 테이블 --- #

        # --- 모델 가져오기, 결과 내보내기 --- #
        self.ui.import_button.clicked.connect(self.importModel)
        self.ui.save_status_button.clicked.connect(self.saveStatus)
        self.ui.save_label_button.clicked.connect(self.saveLabel)
        self.ui.save_json_button.clicked.connect(self.saveJson)
        self.ui.save_button.clicked.connect(self.saveResult)
        self.ui.save_button.setEnabled(False)
        self.ui.save_json_button.hide()
        # --- 모델 가져오기, 결과 내보내기 --- #

        # --- 영상, 사진 미리보기 --- #
        self.ui.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 영상, 사진 미리보기 --- #

        # --- 상태 표시줄 초기화 --- #
        # 상태 표시줄 그림자 효과
        self.shadowStyle(self.ui.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.classesLabel, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.targetLabel, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.fpsLabel, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.modelLabel, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 상태 표시줄 그림자 효과
        self.model_name = self.ui.model_box.currentText()  # model 기본값 가져오기
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
        self.modelnamethreshold(self.ui.Model_label,self.model_name)
        # --- 상태 표시줄 초기화 --- #

        self.initThreads()

        # --- 하이퍼파라미터 조정 --- #
        self.ui.iou_spinbox.valueChanged.connect(
            lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 하이퍼파라미터 조정 --- #

        # --- 시작/중지 --- #
        self.ui.left_skip_button.clicked.connect(self.runLeftSkip)
        self.ui.left_move_button.clicked.connect(self.runLeftMove)
        self.ui.right_move_button.clicked.connect(self.runRightMove)
        self.ui.right_skip_button.clicked.connect(self.runRightSkip)
        self.ui.run_button.clicked.connect(self.runorContinue)
        self.ui.stop_button.clicked.connect(self.stopDetect)
        # --- 시작/중지 --- #

        self.visibleNavigation(False,  self.view_mode)

        # --- 설정 표시줄 초기화 --- #
        self.loadConfig()
        # --- 설정 표시줄 초기화 --- #

        # --- MessageBar Init --- #
        self.showStatus("Welcome to YoloView")
        # --- MessageBar Init --- #

    def initThreads(self):
        self.yolo_threads = YOLOThreadPool()
        # 현재 모델 설정
        model_name = self.checkCurrentModel()
        if model_name:
            self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
            self.initModel(yoloname=model_name)

    def selectedTrackMode(self):
        model_name = self.checkCurrentModel()
        yolo_thread = self.yolo_threads.get(model_name)
        self.updateTrackMode(yolo_thread)

    #결과 내보내기
    def saveResult(self):
        if not any(thread.res_status for thread in self.yolo_threads.threads_pool.values()):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        save_path = config.get('save_path', os.getcwd())
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 부모 창 객체
                "Save Results in new Folder",  # 제목
                save_path,  #시작 디렉토리
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 부모 창 객체
                "Save Image/Video",  # 제목
                save_path,  #시작 디렉토리
                "Image/Vide Type (*.jpg *.jpeg *.png *.heic *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 유형 필터 항목을 선택하고 필터 내용은 괄호 안에 포함.
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=False)

        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)


    # pt 모델을 model_box에 로드
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        # yolov8/yolo11 이면 track 모드 UI 활성화
        if not self.model_initialized_trackmodel:
            self.showTrackStatus(self.model_name)
            self.model_initialized_trackmodel = True # Track 모드 지원 여부 1번만 체크

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box.clear()
            self.ui.model_box.addItems(self.pt_list)

    def stopOtherModelProcess(self, model_name, current_yoloname):
        yolo_thread = self.yolo_threads.get(model_name)
        yolo_thread.finished.connect(lambda: self.resignModel(current_yoloname))
        yolo_thread.stop_dtc = True
        yolo_thread.force_stop_dtc = True
        self.yolo_threads.stop_thread(model_name)

    # 다른 모델 중지
    def stopOtherModel(self, current_yoloname=None):
        for model_name in self.yolo_threads.threads_pool.keys():
            if not current_yoloname or model_name == current_yoloname:
                continue
            if self.yolo_threads.get(model_name).isRunning():
                self.stopOtherModelProcess(model_name, current_yoloname)

    # 모델 다시 로드
    def resignModel(self, model_name):
        glo.set_value('yoloname', model_name)
        self.reloadModel()
        self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
        self.initModel(yoloname=model_name)
        self.runModel(True)

    # 모델 변경
    def changeModel(self, categories_reset = True):
        self.model_name = self.ui.model_box.currentText()
        self.modelnamethreshold(self.ui.Model_label,self.model_name)  # 상태 표시줄 표시 수정
        current_model_name = self.checkCurrentModel()
        if not current_model_name:
            return
        # 다른 모델 중지
        self.stopOtherModel(current_model_name)
        yolo_thread = self.yolo_threads.get(current_model_name)
        if yolo_thread is not None:
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
            if categories_reset:
                self.loadCategories(yolo_thread, 'single')
        else:
            self.yolo_threads.set(current_model_name, MODEL_THREAD_CLASSES[current_model_name]())
            self.initModel(yoloname=current_model_name)
            self.loadConfig()
            self.showStatus(f"Change Model to {current_model_name} Successfully")

        self.showTrackStatus(self.model_name)

    def runModelProcess(self, yolo_name):
        yolo_thread = self.yolo_threads.get(yolo_name)
        self.updateCategories(yolo_thread, 'single')
        # 기존 self.inputPath가 전체 이미지 리스트라면 현재 인덱스에 해당하는 이미지 1장만 처리하도록 source를 변경
        if hasattr(self, 'current_index') and self.checkedNavigationButton():
            current_source = [self.inputPath[self.current_index]]
        else:
            # current_index 미정이면 그냥 전체 리스트 사용 (초기 상태)
            current_source = self.inputPath

        yolo_thread.source = current_source
        yolo_thread.stop_dtc = False
        yolo_thread.force_stop_dtc = False

        if self.ui.run_button.isChecked() or self.checkedNavigationButton():
            yolo_thread.is_continue = True
            self.yolo_threads.start_thread(yolo_name)
        else:
            yolo_thread.is_continue = False
            self.showStatus('Pause Detection')

    def runModel(self, runbuttonStatus=None):
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus:
            self.ui.run_button.setChecked(True)
        current_model_name = self.checkCurrentModel()
        if current_model_name is not None:
            self.runModelProcess(current_model_name)
        else:
            self.showStatus('The current model is not supported')
            if self.ui.run_button.isChecked():
                self.ui.run_button.setChecked(False)

        # 모델 실행 후 버튼 상태 업데이트
        self.updateNavigationButtons()
        self.initNavigationButtons()

    def runorContinue(self):
        self.closeTableResult()
        if self.inputPath is not None and len(self.inputPath) > 0:
            self.changeModel(categories_reset=False)
            self.runModel()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.run_button.setChecked(False)

        # 실행 후 버튼 상태 업데이트
        self.updateNavigationButtons()

    def runLeftSkip(self):
        self.stopDetect()
        if self.inputPath is not None and isinstance(self.inputPath, list) and len(self.inputPath) > 0:
            self.ui.left_skip_button.setChecked(True)
            # 첫 번째 이미지로 이동
            self.current_index = 0
            self.runorContinue()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.left_skip_button.setChecked(False)

        # 이동 후 버튼 상태 업데이트
        self.updateNavigationButtons()

    def runLeftMove(self):
        self.stopDetect()
        if self.inputPath is not None and isinstance(self.inputPath, list) and len(self.inputPath) > 0:
            self.ui.left_move_button.setChecked(True)
            # 현재 인덱스가 0보다 클 때만 왼쪽으로 이동
            if self.current_index > 0:
                self.current_index -= 1
                self.runorContinue()
            else:
                self.showStatus("Already at the first image.")
                self.ui.left_move_button.setChecked(False)
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.left_move_button.setChecked(False)

        # 이동 후 버튼 상태 업데이트
        self.updateNavigationButtons()

    def runRightMove(self):
        self.stopDetect()
        if self.inputPath is not None and isinstance(self.inputPath, list) and len(self.inputPath) > 0:
            self.ui.right_move_button.setChecked(True)
            # 현재 인덱스가 마지막 인덱스보다 작을 때 오른쪽으로 이동
            if self.current_index < len(self.inputPath) - 1:
                self.current_index += 1
                self.runorContinue()
            else:
                self.showStatus("Already at the last image.")
                self.ui.right_move_button.setChecked(False)
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.right_move_button.setChecked(False)

        # 이동 후 버튼 상태 업데이트
        self.updateNavigationButtons()

    def runRightSkip(self):
        self.stopDetect()
        if self.inputPath is not None and isinstance(self.inputPath, list) and len(self.inputPath) > 0:
            self.ui.right_skip_button.setChecked(True)
            # 마지막 이미지로 이동
            self.current_index = len(self.inputPath) - 1
            self.runorContinue()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.right_skip_button.setChecked(False)

        # 이동 후 버튼 상태 업데이트
        self.updateNavigationButtons()

    def updateNavigationButtons(self):
        # inputPath가 존재하고, 그 길이가 1 이상일 때만 버튼 활성화 로직을 진행
        if self.inputPath and len(self.inputPath) > 0:
            # 첫 번째 이미지인지 판단
            is_first_image = (self.current_index == 0)
            # 마지막 이미지인지 판단
            is_last_image = (self.current_index == len(self.inputPath) - 1)

            # 첫 번째 이미지일 경우 '처음으로', '이전' 버튼 비활성화
            self.ui.left_skip_button.setEnabled(not is_first_image)
            self.ui.left_move_button.setEnabled(not is_first_image)

            # 마지막 이미지일 경우 '다음', '마지막으로' 버튼 비활성화
            self.ui.right_move_button.setEnabled(not is_last_image)
            self.ui.right_skip_button.setEnabled(not is_last_image)
        else:
            # inputPath가 비어있거나 없을 경우 모든 이동 관련 버튼 비활성화
            self.ui.left_skip_button.setEnabled(False)
            self.ui.left_move_button.setEnabled(False)
            self.ui.right_move_button.setEnabled(False)
            self.ui.right_skip_button.setEnabled(False)

    def initNavigationButtons(self):
        # **여기 추가: 네비게이션 버튼 체크 상태 초기화**
        self.ui.left_skip_button.setChecked(False)
        self.ui.right_skip_button.setChecked(False)
        self.ui.left_move_button.setChecked(False)
        self.ui.right_move_button.setChecked(False)

    def checkedNavigationButton(self):
        is_check = (self.ui.left_skip_button.isChecked() or self.ui.right_skip_button.isChecked() or
                    self.ui.left_move_button.isChecked() or self.ui.right_move_button.isChecked())
        return is_check

    # 인식 중지
    def stopDetect(self):
        self.closeTableResult()
        self.quitRunningModel(stop_status=True)
        self.ui.run_button.setChecked(False)
        self.ui.save_status_button.setEnabled(True)
        self.ui.progress_bar.setValue(0)
        self.ui.main_leftbox.clear()  # clear image display
        self.ui.main_rightbox.clear()
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
