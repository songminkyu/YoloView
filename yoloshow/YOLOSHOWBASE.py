from ui.utils.AcrylicFlyout import AcrylicFlyoutView, AcrylicFlyout
from ui.utils.TableView import TableViewQWidget
from ui.utils.drawFigure import PlottingThread
from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov8 yolov9 yolov9-seg yolov10 yolo11 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")

from utils.logger import LoggerUtils
import re
import socket
from urllib.parse import urlparse
import torch
import json
import os
import shutil
import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFileDialog, QGraphicsDropShadowEffect, QFrame, QPushButton, QApplication
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, \
    QParallelAnimationGroup, QPoint
from qfluentwidgets import RoundMenu, MenuAnimationType, Action
import importlib
from ui.utils.rtspDialog import CustomMessageBox

from models import common, yolo, experimental
from ui.utils.webCamera import Camera, WebcamThread

GLOBAL_WINDOW_STATE = True

WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_SETTING_BAR = 300
WIDTH_LOGO = 60
WINDOW_SPLIT_BODY = 20
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']
ALL_MODEL_NAMES = ["yolov5", "yolov8", "yolov9", "yolov9-seg","yolo11", "yolov5-seg", "yolov8-seg", "rtdetr", "yolov8-pose"]

loggertool = LoggerUtils()


# YOLOSHOW 윈도우 클래스는 UI 파일과 Ui_mainWindow를 동적으로 로드합니다.
class YOLOSHOWBASE:
    def __init__(self):
        super().__init__()
        self.inputPath = None
        self.yolo_threads = None
        self.result_statistic = None
        self.detect_result = None

    # 왼쪽 메뉴바 초기화
    def initSiderWidget(self):
        # --- 사이드바 --- #
        self.ui.leftBox.setFixedWidth(WIDTH_LEFT_BOX_STANDARD)
        # logo
        self.ui.logo.setFixedSize(WIDTH_LOGO, WIDTH_LOGO)

        # 왼쪽 메뉴 표시줄의 버튼 너비 수정
        for child_left_box_widget in self.ui.leftbox_bottom.children():

            if isinstance(child_left_box_widget, QFrame):
                child_left_box_widget.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

                for child_left_box_widget_btn in child_left_box_widget.children():
                    if isinstance(child_left_box_widget_btn, QPushButton):
                        child_left_box_widget_btn.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

    # 모델 로드
    def initModel(self, model_thread, yoloname=None):
        model_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        model_thread.progress_value = self.ui.progress_bar.maximum()
        model_thread.send_input.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        model_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
        model_thread.send_msg.connect(lambda x: self.showStatus(x))
        model_thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
        model_thread.send_fps.connect(lambda x: self.ui.fps_label.setText(str(x)))
        model_thread.send_class_num.connect(lambda x: self.ui.Class_num.setText(str(x)))
        model_thread.send_target_num.connect(lambda x: self.ui.Target_num.setText(str(x)))
        model_thread.send_result_picture.connect(lambda x: self.setResultStatistic(x))
        model_thread.send_result_table.connect(lambda x: self.setTableResult(x))

        return model_thread

    def updateTrackMode(self,model_thread,yoloname=None):
        model_thread.track_mode = True if self.ui.track_box.currentText() == "On" else False

    # 그림자 효과
    def shadowStyle(self, widget, Color, top_bottom=None):
        shadow = QGraphicsDropShadowEffect(self)
        if 'top' in top_bottom and 'bottom' not in top_bottom:
            shadow.setOffset(0, -5)
        elif 'bottom' in top_bottom and 'top' not in top_bottom:
            shadow.setOffset(0, 5)
        else:
            shadow.setOffset(5, 5)
        shadow.setBlurRadius(10)  # 그림자 반경
        shadow.setColor(Color)  # 그림자 색상
        widget.setGraphicsEffect(shadow)

    # 사이드바 확대/축소
    def scaleMenu(self):
        # standard = 80
        # maxExtend = 180

        leftBoxStart = self.ui.leftBox.width()
        _IS_EXTENDED = leftBoxStart == WIDTH_LEFT_BOX_EXTENDED

        if _IS_EXTENDED:
            leftBoxEnd = WIDTH_LEFT_BOX_STANDARD
        else:
            leftBoxEnd = WIDTH_LEFT_BOX_EXTENDED

        # animation
        self.animation = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.animation.setDuration(500)  # ms
        self.animation.setStartValue(leftBoxStart)
        self.animation.setEndValue(leftBoxEnd)
        self.animation.setEasingCurve(QEasingCurve.InOutQuint)
        self.animation.start()

    # 막대 확대/축소 설정
    def scalSetting(self):
        # GET WIDTH
        widthSettingBox = self.ui.settingBox.width()  # right set column width
        widthLeftBox = self.ui.leftBox.width()  # left column length
        maxExtend = WIDTH_SETTING_BAR
        standard = 0

        # SET MAX WIDTH
        if widthSettingBox == 0:
            widthExtended = maxExtend
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                    background-color: qlineargradient(x1:0, y1:0, x2:1 , y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
                                }
                              """)
        else:
            widthExtended = standard
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                }
                              """)

        # ANIMATION LEFT BOX
        self.left_box = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.left_box.setDuration(500)
        self.left_box.setStartValue(widthLeftBox)
        self.left_box.setEndValue(68)
        self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

        # ANIMATION SETTING BOX
        self.setting_box = QPropertyAnimation(self.ui.settingBox, b"minimumWidth")
        self.setting_box.setDuration(500)
        self.setting_box.setStartValue(widthSettingBox)
        self.setting_box.setEndValue(widthExtended)
        self.setting_box.setEasingCurve(QEasingCurve.InOutQuart)

        # SET QSS Change
        self.qss_animation = QPropertyAnimation(self.ui.mainbox, b"styleSheet")
        self.qss_animation.setDuration(300)
        self.qss_animation.setStartValue("""
            QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
            }
        """)
        self.qss_animation.setEndValue("""
             QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
            }
        """)
        self.qss_animation.setEasingCurve(QEasingCurve.InOutQuart)

        # GROUP ANIMATION
        self.group = QParallelAnimationGroup()
        self.group.addAnimation(self.left_box)
        self.group.addAnimation(self.setting_box)
        self.group.start()

    # 최소화 창 최대화
    def maxorRestore(self):
        global GLOBAL_WINDOW_STATE
        status = GLOBAL_WINDOW_STATE
        if status:
            # 현재 화면의 너비와 높이를 가져옴.
            self.showMaximized()
            self.ui.maximizeButton.setStyleSheet("""
                          QPushButton:hover{
                               background-color:rgb(139, 29, 31);
                               border-image: url(:/leftbox/images/newsize/scalling.png);
                           }
                      """)
            GLOBAL_WINDOW_STATE = False
        else:
            self.showNormal()
            self.ui.maximizeButton.setStyleSheet("""
                                      QPushButton:hover{
                                           background-color:rgb(139, 29, 31);
                                           border-image: url(:/leftbox/images/newsize/max.png);
                                       }
                                  """)
            GLOBAL_WINDOW_STATE = True

    # 사진/비디오 선택 및 표시
    def selectFile(self):
        # 마지막으로 선택한 파일의 경로를 가져옴.
        config_file = f'{self.current_workpath}/config/file.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_path = config['file_path']
        if not os.path.exists(file_path):
            file_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 부모 창 객체
            "Select your Image / Video",  # 제목
            file_path,  # 기본 열기 경로는 현재 경로
            "Image / Video type (*.jpg *.jpeg *.png *.bmp *.dib *.jpe *.jp2 *.mp4)"  # 유형 필터 항목을 선택하고 필터 내용은 괄호 안에 있습니다.
        )
        if file:
            self.inputPath = file
            glo.set_value('inputPath', self.inputPath)
            # 비디오인 경우 첫 번째 프레임을 표시.
            if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                # 첫 번째 프레임 표시
                self.cap = cv2.VideoCapture(self.inputPath)
                ret, frame = self.cap.read()
                if ret:
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.showImg(frame, self.ui.main_leftbox, 'img')
            # 사진이라면 정상적으로 표시 됨.
            else:
                self.showImg(self.inputPath, self.ui.main_leftbox, 'path')
            self.showStatus('Loaded File：{}'.format(os.path.basename(self.inputPath)))
            config['file_path'] = os.path.dirname(self.inputPath)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 카메라 선택
    def selectWebcam(self):
        try:
            # get the number of local cameras
            cam_num, cams = Camera().get_cam_num()
            if cam_num > 0:
                popMenu = RoundMenu(parent=self)
                popMenu.setFixedWidth(self.ui.leftbox_bottom.width())
                actions = []

                for cam in cams:
                    cam_name = f'Camera_{cam}'
                    actions.append(Action(cam_name))
                    popMenu.addAction(actions[-1])
                    actions[-1].triggered.connect(lambda: self.actionWebcam(cam))

                x = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).x()
                y = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).y()
                y = y - self.ui.webcamBox.frameGeometry().height() * 2
                pos = QPoint(x, y)
                popMenu.exec(pos, aniType=MenuAnimationType.DROP_DOWN)
            else:
                self.showStatus('No camera found !!!')
        except Exception as e:
            self.showStatus('%s' % e)

    # 웹캠 연결
    def actionWebcam(self, cam):
        self.showStatus(f'Loading camera：Camera_{cam}')
        self.thread = WebcamThread(cam)
        self.thread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        self.thread.start()
        self.inputPath = int(cam)

    # 폴더 선택
    def selectFolder(self):
        config_file = f'{self.current_workpath}/config/folder.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        folder_path = config['folder_path']
        if not os.path.exists(folder_path):
            folder_path = os.getcwd()
        FolderPath = QFileDialog.getExistingDirectory(
            self,
            "Select your Folder",
            folder_path  # 시작 디렉토리
        )
        if FolderPath:
            FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            # self.yolov5_thread.source = Foldername
            self.inputPath = Foldername
            self.showStatus('Loaded Folder：{}'.format(os.path.basename(FolderPath)))
            config['folder_path'] = FolderPath
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 웹캠 Rtsp 선택
    def selectRtsp(self):
        # rtsp://rtsp-test-server.viomic.com:554/stream
        rtspDialog = CustomMessageBox(self, mode="single")
        self.rtspUrl = None
        if rtspDialog.exec():
            self.rtspUrl = rtspDialog.urlLineEdit.text()
        if self.rtspUrl:
            parsed_url = urlparse(self.rtspUrl)
            if parsed_url.scheme == 'rtsp':
                if not self.checkRtspUrl(self.rtspUrl):
                    self.showStatus('Rtsp stream is not available')
                    return False
                self.showStatus(f'Loading Rtsp：{self.rtspUrl}')
                self.rtspThread = WebcamThread(self.rtspUrl)
                self.rtspThread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
                self.rtspThread.start()
                self.inputPath = self.rtspUrl
            elif parsed_url.scheme in ['http', 'https']:
                if not self.checkHttpUrl(self.rtspUrl):
                    self.showStatus('Http stream is not available')
                    return False
                self.showStatus(f'Loading Http：{self.rtspUrl}')
                self.rtspThread = WebcamThread(self.rtspUrl)
                self.rtspThread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
                self.rtspThread.start()
                self.inputPath = self.rtspUrl
            else:
                self.showStatus('URL is not an rtsp stream')
                return False

    # 웹캠 Rtsp가 연결되어 있는지 확인
    def checkRtspUrl(self, url, timeout=5):
        try:
            # URL을 구문 분석하여 호스트 이름과 포트를 취득
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 554  # RTSP 기본 포트는 554

            # socket 객체 생성
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # socket 연결 시도
            sock.connect((hostname, port))
            # socket 닫기
            sock.close()
            return True
        except Exception as e:
            return False

    # HTTP 웹캠이 연결되어 있는지 확인
    def checkHttpUrl(self, url, timeout=5):
        try:
            # URL을 구문 분석하여 호스트 이름과 포트를 취득
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 80  # HTTP 기본 포트는 80

            # socket 객체 생성
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # socket 연결 시도
            sock.connect((hostname, port))
            # socket 닫기
            sock.close()
            return True
        except Exception as e:
            return False

    # 라벨 이미지 표시
    def showImg(self, img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    # 창 크기 조정
    def resizeGrip(self):
        self.left_grip.setGeometry(0, 10, 10, self.height())
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 모듈 가져오기
    def importModel(self):
        # 마지막으로 선택한 파일의 경로를 가져옴.
        config_file = f'{self.current_workpath}/config/model.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.model_path = config['model_path']
        if not os.path.exists(self.model_path):
            self.model_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 부모 창 객체
            "Select your YOLO Model",  # 标题
            self.model_path,  # 기본 열기 경로는 현재 경로.
            "Model File (*.pt)"  # 유형 필터 항목을 선택하고 필터 내용은 괄호 안에 포함.
        )
        if file:
            fileptPath = os.path.join(self.pt_Path, os.path.basename(file))
            if not os.path.exists(fileptPath):
                shutil.copy(file, self.pt_Path)
                self.showStatus('Loaded Model：{}'.format(os.path.basename(file)))
                config['model_path'] = os.path.dirname(file)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            else:
                self.showStatus('Model already exists')

    # Modelname의 세그먼트 이름 지정 문제 해결
    def checkSegName(self, modelname):
        if "yolov5" in modelname:
            return bool(re.match(r'yolov5.?-seg.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(r'yolov8.?-seg.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(r'yolov9.?-seg.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(r'yolov10.?-seg.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(r'yolo11.?-seg.*\.pt$', modelname))

    # Modelname의 포즈 명명 문제 해결
    def checkPoseName(self, modelname):
        if "yolov5" in modelname:
            return bool(re.match(r'yolov5.?-pose.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(r'yolov8.?-pose.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(r'yolov9.?-pose.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(r'yolov10.?-pose.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(r'yolo11.?-pose.*\.pt$', modelname))

    # Modelname의 포즈 명명 문제 해결
    def checkObbName(self, modelname):
        if "yolov5" in modelname:
            return bool(re.match(r'yolov5.?-obb.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(r'yolov8.?-obb.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(r'yolov9.?-obb.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(r'yolov10.?-obb.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(r'yolo11.?-obb.*\.pt$', modelname))

    # 실행 중인 모델 중지
    def quitRunningModel(self, stop_status=False):
        self.initThreads()
        for yolo_thread in self.yolo_threads:
            try:
                if yolo_thread.isRunning():
                    yolo_thread.quit()
                if stop_status:
                    yolo_thread.stop_dtc = True
            except Exception as err:
                loggertool.info(f"Error: {err}")
                pass

    # MessageBar에 메시지 표시
    def showStatus(self, msg):
        self.ui.message_bar.setText(msg)
        if msg == 'Finish Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.progress_bar.setValue(0)
            self.ui.save_status_button.setEnabled(True)
        elif msg == 'Stop Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.save_status_button.setEnabled(True)
            self.ui.progress_bar.setValue(0)
            self.ui.main_leftbox.clear()  # clear image display
            self.ui.main_rightbox.clear()
            self.ui.Class_num.setText('--')
            self.ui.Target_num.setText('--')
            self.ui.fps_label.setText('--')

    # 내보내기 결과 상태
    def saveStatus(self):
        if self.ui.save_status_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run image results are not saved.')
            for yolo_thread in self.yolo_threads:
                yolo_thread.save_res = False
            self.ui.save_button.setEnabled(False)
        elif self.ui.save_status_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run image results will be saved.')
            for yolo_thread in self.yolo_threads:
                yolo_thread.save_res = True
            self.ui.save_button.setEnabled(True)

    # 결과 내보내기
    def saveResult(self):
        if (not self.yolov5_thread.res_status and not self.yolov7_thread.res_status
                and not self.yolov8_thread.res_status and not self.yolov9_thread.res_status
                and not self.yolov9seg_thread.res_status and not self.yolov5seg_thread.res_status
                and not self.yolov8seg_thread.res_status and not self.rtdetr_thread.res_status
                and not self.yolov8pose_thread.res_status and not self.yolov11_thread.res_status
                and not self.yolov11pose_thread.res_status and not self.yolov11seg_thread.res_status):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        save_path = config['save_path']
        if not os.path.exists(save_path):
            save_path = os.getcwd()
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 부모 창 객체
                "Save Results in new Folder",  # 제목
                save_path,  # 시작 디렉토리
            )
            if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5_thread, folder=True)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread, folder=True)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread, folder=True)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread, folder=True)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread, folder=True)
            elif "yolov9" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov9_thread, folder=True)
            elif "yolov9" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov9seg_thread, folder=True)
            elif "yolo11" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11_thread, folder=True)
            elif "yolo11" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11seg_thread, folder=True)
            elif "yolo11" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11pose_thread, folder=True)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 부모 창 객체
                "Save Image/Video",  # 제목
                save_path,  # 시작 디렉토리
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 유형 필터 항목을 선택하고 필터 내용은 괄호 안에 포함.
            )
            if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5_thread, folder=False)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread, folder=False)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread, folder=False)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread, folder=False)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread, folder=False)
            elif "yolov9" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov9_thread, folder=False)
            elif "yolov9" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov9seg_thread, folder=False)
            elif "yolo11" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11_thread, folder=False)
            elif "yolo11" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11seg_thread, folder=False)
            elif "yolo11" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov11pose_thread, folder=False)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread, folder=False)

        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 테스트 결과 내보내기 --- 프로세스 코드
    def saveResultProcess(self, outdir, yolo_thread, folder):
        if folder:
            try:
                output_dir = os.path.dirname(yolo_thread.res_path)
                if os.path.exists(output_dir):
                    for filename in os.listdir(output_dir):
                        source_path = os.path.join(output_dir, filename)
                        destination_path = os.path.join(outdir, filename)
                        if os.path.isfile(source_path):
                            shutil.copy(source_path, destination_path)
                    self.showStatus('Saved Successfully in {}'.format(outdir))
                else:
                    self.showStatus('Please wait for the result to be generated')
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")
        else:
            try:
                if os.path.exists(yolo_thread.res_path):
                    shutil.copy(yolo_thread.res_path, outdir)
                    self.showStatus('Saved Successfully in {}'.format(outdir))
                else:
                    self.showStatus('Please wait for the result to be generated')
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")

    # 로드 설정 표시줄
    def loadConfig(self):
        config_file = self.current_workpath + '/config/setting.json'
        iou = 0.45
        conf = 0.25
        delay = 10
        line_thickness = 3
        if not os.path.exists(config_file):
            iou = 0.45
            conf = 0.25
            delay = 10
            line_thickness = 3
            new_config = {"iou": iou,
                          "conf": conf,
                          "delay": delay,
                          "line_thickness": line_thickness,
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 4:
                iou = 0.45
                conf = 0.25
                delay = 10
                line_thickness = 3
            else:
                iou = config['iou']
                conf = config['conf']
                delay = config['delay']
                line_thickness = config['line_thickness']
        self.ui.iou_spinbox.setValue(iou)
        self.ui.iou_slider.setValue(int(iou * 100))
        self.ui.conf_spinbox.setValue(conf)
        self.ui.conf_slider.setValue(int(conf * 100))
        self.ui.speed_spinbox.setValue(delay)
        self.ui.speed_slider.setValue(delay)
        self.ui.line_spinbox.setValue(line_thickness)
        self.ui.line_slider.setValue(line_thickness)

    # pt 모델을 model_box에 로드
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box.clear()
            self.ui.model_box.addItems(self.pt_list)

    # 모델 다시 로드
    def reloadModel(self):
        importlib.reload(common)
        importlib.reload(yolo)
        importlib.reload(experimental)

    # 하이퍼파라미터 튜닝
    def changeValue(self, x, flag):
        if flag == 'iou_spinbox':
            self.ui.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.ui.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.showStatus('IOU Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads:
                yolo_thread.iou_thres = x / 100
            # self.yolov5_thread.iou_thres = x / 100
            # self.yolov7_thread.iou_thres = x / 100
            # self.yolov8_thread.iou_thres = x / 100
            # self.yolov9_thread.iou_thres = x / 100
            # self.yolov11_thread.iou_thres = x / 100
            # self.yolov5seg_thread.iou_thres = x / 100
            # self.yolov8seg_thread.iou_thres = x / 100
            # self.rtdetr_thread.iou_thres = x / 100
            # self.yolov8pose_thread.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.ui.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.ui.conf_spinbox.setValue(x / 100)
            self.showStatus('Conf Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads:
                yolo_thread.conf_thres = x / 100
            # self.yolov5_thread.conf_thres = x / 100
            # self.yolov7_thread.conf_thres = x / 100
            # self.yolov8_thread.conf_thres = x / 100
            # self.yolov9_thread.conf_thres = x / 100
            # self.yolov11_thread.conf_thres = x / 100
            # self.yolov5seg_thread.conf_thres = x / 100
            # self.yolov8seg_thread.conf_thres = x / 100
            # self.rtdetr_thread.conf_thres = x / 100
            # self.yolov8pose_thread.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.ui.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.ui.speed_spinbox.setValue(x)
            self.showStatus('Delay: %s ms' % str(x))
            for yolo_thread in self.yolo_threads:
                yolo_thread.speed_thres = x  # ms
            # self.yolov5_thread.speed_thres = x  # ms
            # self.yolov7_thread.speed_thres = x  # ms
            # self.yolov8_thread.speed_thres = x  # ms
            # self.yolov9_thread.speed_thres = x  # ms
            # self.yolov11_thread.speed_thres = x / ms
            # self.yolov5seg_thread.speed_thres = x  # ms
            # self.yolov8seg_thread.speed_thres = x  # ms
            # self.rtdetr_thread.speed_thres = x  # ms
            # self.yolov8pose_thread.speed_thres = x  # ms
        elif flag == 'line_spinbox':
            self.ui.line_slider.setValue(x)
        elif flag == 'line_slider':
            self.ui.line_spinbox.setValue(x)
            self.showStatus('Line Width: %s' % str(x))
            for yolo_thread in self.yolo_threads:
                yolo_thread.line_thickness = x
            # self.yolov5_thread.line_thickness = x
            # self.yolov7_thread.line_thickness = x
            # self.yolov8_thread.line_thickness = x
            # self.yolov9_thread.line_thickness = x
            # self.yolov11_thread.line_thickness = x
            # self.yolov5seg_thread.line_thickness = x
            # self.yolov8seg_thread.line_thickness = x
            # self.rtdetr_thread.line_thickness = x
            # self.yolov8pose_thread.line_thickness = x

    # YOLOv5, YOLOv7 및 YOLOv9를 수정하여 yolo.py 충돌 해결
    def solveYoloConflict(self, ptnamelst):
        glo.set_value("yoloname", "yolov5 yolov7 yolov8 yolov9 yolov9-seg yolov5-seg yolov8-seg rtdetr yolov8-pose yolo11 yolo11-seg yolo11-pose")
        self.reloadModel()

    # 통계 결과를 수락하고 json에 기록
    def setResultStatistic(self, value):
        # JSON 파일에 쓰기
        with open('config/result.json', 'w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=4)
        # --- 통계 결과 얻기 + 히스토그램 그리기 --- #
        self.result_statistic = value
        self.plot_thread = PlottingThread(self.result_statistic, self.current_workpath)
        self.plot_thread.start()
        # --- 통계 결과 얻기 + 히스토그램 그리기 --- #

    # 막대 차트 결과 표시
    def showResultStatics(self):
        self.resutl_statistic = dict()
        # JSON 파일 읽기
        with open(self.current_workpath + r'\config\result.json', 'r', encoding='utf-8') as file:
            self.result_statistic = json.load(file)
        if self.result_statistic:
            # 한국어 키를 사용하여 새 사전 만들기
            result_str = ""
            for index, (key, value) in enumerate(self.result_statistic.items()):
                result_str += f"{key}:{value}x \t"
                if (index + 1) % 4 == 0:
                    result_str += "\n"

            view = AcrylicFlyoutView(
                title='Detected Target Category Distribution (Percentage)',
                content=result_str,
                image=self.current_workpath + r'\config\result.png',
                isClosable=True
            )

        else:
            view = AcrylicFlyoutView(
                title='Result Statistics',
                content="No completed target detection results detected, please execute the detection task first!",
                isClosable=True
            )

        # 글꼴 크기 수정
        view.titleLabel.setStyleSheet("""font-size: 30px; 
                                            color: black; 
                                            font-weight: bold; 
                                            font-family: 'KaiTi';
                                        """)
        view.contentLabel.setStyleSheet("""font-size: 25px; 
                                            color: black; 
                                            font-family: 'KaiTi';""")
        # 이미지 크기 수정
        width = self.ui.rightbox_main.width() // 2.5
        height = self.ui.rightbox_main.height() // 2.5
        view.imageLabel.setFixedSize(width, height)
        # adjust layout (optional)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)

        # show view
        w = AcrylicFlyout.make(view, self.ui.rightbox_play, self)
        view.closed.connect(w.close)

    # 테이블 결과 목록 가져오기
    def setTableResult(self, value):
        self.detect_result = value

    # 테이블 결과 표시
    def showTableResult(self):
        self.table_result = TableViewQWidget(infoList=self.detect_result)
        self.table_result.show()
