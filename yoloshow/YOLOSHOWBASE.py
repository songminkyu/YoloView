import random
from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov8 yolov9 yolov9-seg yolov10 yolo11 "
                            "yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb "
                            "fastsam sam samv2 bbox-valid seg-valid ")
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
from ui.utils.AcrylicFlyout import ResultChartView
from ui.utils.TableView import TableViewQWidget
from ui.utils.drawFigure import PlotWindow
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFileDialog, QGraphicsDropShadowEffect, QFrame, QPushButton
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QPoint
from qfluentwidgets import RoundMenu, MenuAnimationType, Action
import importlib
from ui.utils.rtspDialog import RtspInputMessageBox
from ui.utils.CustomMessageBox import MessageBox

from models import common
from ui.utils.webCamera import Camera, WebcamThread

from yolocode.YOLOv8Thread import YOLOv8Thread
from yolocode.YOLOv8SegThread import YOLOv8SegThread
from yolocode.RTDETRThread import RTDETRThread
from yolocode.YOLOv8PoseThread import YOLOv8PoseThread
from yolocode.YOLOv8ObbThread import YOLOv8ObbThread
from yolocode.FastSAMThread import FastSAMThread
from yolocode.SAMThread import SAMThread
from yolocode.SAMv2Thread import SAMv2Thread
from yolocode.BBoxValidThread import BBoxValidThread
from yolocode.SegValidThread import SegValidThread
from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics import RTDETR

GLOBAL_WINDOW_STATE = True

WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_SETTING_BAR = 300
WIDTH_LOGO = 60
WINDOW_SPLIT_BODY = 20
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']
# 모델 이름 및 스레드 클래스 매핑
MODEL_THREAD_CLASSES = {
    "yolov5": YOLOv8Thread,
    "yolov5-seg": YOLOv8SegThread,
    "yolov9": YOLOv8Thread,
    "yolov10": YOLOv8Thread,
    "rtdetr": RTDETRThread,
    "yolov8": YOLOv8Thread,
    "yolov8-seg": YOLOv8SegThread,
    "yolov8-pose": YOLOv8PoseThread,
    "yolov8-obb": YOLOv8ObbThread,
    "yolov11": YOLOv8Thread,
    "yolov11-seg": YOLOv8SegThread,
    "yolov11-obb": YOLOv8ObbThread,
    "yolov11-pose": YOLOv8PoseThread,
    "fastsam": FastSAMThread,
    "sam": SAMThread,
    "samv2": SAMv2Thread,
    "bbox-valid": BBoxValidThread,
    "seg-valid": SegValidThread,
}
# 扩展MODEL_THREAD_CLASSES字典
MODEL_NAME_DICT = list(MODEL_THREAD_CLASSES.items())
for key, value in MODEL_NAME_DICT:
    MODEL_THREAD_CLASSES[f"{key}_left"] = value
    MODEL_THREAD_CLASSES[f"{key}_right"] = value

ALL_MODEL_NAMES = ["yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov5-seg", "yolov8-seg", "rtdetr",
                   "yolov8-pose", "yolov8-obb","fastsam", "sam", "samv2", "bbox-valid", "seg-valid"]
loggertool = LoggerUtils()


# YOLOSHOW 윈도우 클래스는 UI 파일과 Ui_mainWindow를 동적으로 로드합니다.
class YOLOSHOWBASE:
    def __init__(self):
        super().__init__()
        self.statistic_plot = None
        self.current_workpath = None
        self.inputPath = None
        self.yolo_threads = None
        self.result_statistic = None
        self.detect_result = None
        self.detect_errors = None
        self.table_result = None
        self.allModelNames = ALL_MODEL_NAMES

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
    def initModel(self, yoloname=None):
        thread = self.yolo_threads.get(yoloname)
        if not thread:
            raise ValueError(f"No thread found for '{yoloname}'")
        thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        thread.progress_value = self.ui.progress_bar.maximum()
		# 신호 및 슬롯 연결은 별도로 정의된 기능을 사용하여 클로저 생성을 줄입니다.
        thread.send_input.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
        thread.send_msg.connect(lambda x: self.showStatus(x))
        thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
        thread.send_fps.connect(lambda x: self.ui.fps_label.setText(str(x)))
        thread.send_class_num.connect(lambda x: self.ui.Class_num.setText(str(x)))
        thread.send_target_num.connect(lambda x: self.ui.Target_num.setText(str(x)))
        thread.send_result_picture.connect(lambda x: self.setResultStatistic(x))
        thread.send_result_table.connect(lambda x,y: self.setTableResult(x,y))

        self.loadCategories(thread, 'single')

    def updateTrackMode(self, thread):
        thread.track_mode = True if self.ui.track_box.currentText() == "On" else False

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
            "Image / Video type (*.jpg *.jpeg *.png *.heic *.bmp *.dib *.jpe *.jp2 *.mp4)"  # 유형 필터 항목을 선택하고 필터 내용은 괄호 안에 있습니다.
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
        current_model = self.checkCurrentModel()
        if current_model in ['bbox-valid', 'seg-valid']:
            self.inputPath = [FolderPath]
        else:
            if FolderPath:
                # 하위 디렉토리 존재 여부 확인
                has_subdirectories = any(
                    os.path.isdir(os.path.join(FolderPath, subdir))
                    for subdir in os.listdir(FolderPath)
                )

                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png",
                              ".heic", ".jpeg", ".bmp", ".dib", ".jpe",".jp2"]

                Foldername = []

                # 기본적으로 현재 디렉토리만 탐색
                search_subdirs = False

                if has_subdirectories:
                    # 하위 디렉토리가 있을 경우 메시지 박스 표시
                    msgDialog = MessageBox(
                        self,
                        "Would you like to include subdirectories in the search?",
                        "Please note that searching through subdirectories may take additional time."
                    )
                    if msgDialog.exec():
                        search_subdirs = True

                if search_subdirs:
                    # 하위 디렉토리까지 탐색
                    for root, dirs, files in os.walk(FolderPath):
                        for filename in files:
                            if os.path.splitext(filename)[1].lower() in FileFormat:
                                Foldername.append(os.path.join(root, filename))
                else:
                    # 현재 디렉토리만 탐색
                    for filename in os.listdir(FolderPath):
                        if os.path.splitext(filename)[1].lower() in FileFormat:
                            Foldername.append(os.path.join(FolderPath, filename))

                self.inputPath = Foldername
                self.showStatus('Loaded Folder: {}'.format(os.path.basename(FolderPath)))

        if os.path.exists(FolderPath):
            config['folder_path'] = FolderPath
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 웹캠 Rtsp 선택
    def selectRtsp(self):
        # rtsp://rtsp-test-server.viomic.com:554/stream
        rtspDialog = RtspInputMessageBox(self, mode="single")
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
        except Exception:
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
            "Select your YOLO Model",  # 제목
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

    # 현재 모델 보기
    def checkCurrentModel(self, mode=None):
        # 모델과 해당 조건 간의 매핑 정의
        model_conditions = {
            "yolov5": lambda name: "yolov5" in name and not self.checkSegName(name),
            "yolov8": lambda name: "yolov8" in name and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]),
            "yolov9": lambda name: "yolov9" in name,
            "yolov10": lambda name: "yolov10" in name,
            "yolov11": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]),
            "rtdetr": lambda name: "rtdetr" in name,
            "yolov5-seg": lambda name: "yolov5" in name and self.checkSegName(name),
            "yolov8-seg": lambda name: "yolov8" in name and self.checkSegName(name),
            "yolov11-seg": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkSegName(name),
            "yolov8-pose": lambda name: "yolov8" in name and self.checkPoseName(name),
            "yolov11-pose": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkPoseName(name),
            "yolov8-obb": lambda name: "yolov8" in name and self.checkObbName(name),
            "yolov11-obb": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkObbName(name),
            "fastsam": lambda name: "fastsam" in name,
            "samv2": lambda name: any(sub in name for sub in ["sam2", "samv2"]),
            "sam": lambda name: "sam" in name,
            "bbox-valid": lambda name: "bbox-valid" in name,
            "seg-valid": lambda name: "seg-valid" in name
        }

        if mode:
            # VS mode
            model_name = self.model_name1 if mode == "left" else self.model_name2
            model_name = model_name.lower()
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return f"{yoloname}_{mode}"
        else:
            # Single mode
            model_name = self.model_name.lower()
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return yoloname
        return None

    # 모델이 명명 요구 사항을 충족하는지 확인
    def checkModelName(self, modelname):
        for name in self.allModelNames:
            if modelname in name:
                return True
        return False

    def checkTaskName(self, modelname, taskname):
        if "yolov5" in modelname:
            return bool(re.match(r'yolo.?5.?-' + taskname + r'.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(r'yolo.?8.?-' + taskname + r'.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(r'yolo.?9.?-' + taskname + r'.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(r'yolo.?10.?-' + taskname + r'.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(r'yolo.?11.?-' + taskname + r'.*\.pt$', modelname))

    # Modelname의 세그먼트 이름 지정 문제 해결
    def checkSegName(self, modelname):
        return self.checkTaskName(modelname, "seg")

    # Modelname의 포즈 명명 문제 해결
    def checkPoseName(self, modelname):
        return self.checkTaskName(modelname, "pose")

    # Modelname의 포즈 명명 문제 해결
    def checkObbName(self, modelname):
        return self.checkTaskName(modelname, "obb")

    # 실행 중인 모델 중지
    def quitRunningModel(self, stop_status=False):
        for yolo_name in self.yolo_threads.threads_pool.keys():
            try:
                if stop_status:
                    self.yolo_threads.get(yolo_name).stop_dtc = True
                self.yolo_threads.stop_thread(yolo_name)
            except Exception as err:
                loggertool.info(f"Error: {err}")

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

    def showTrackStatus(self, model_name):
        # 모델명이 yolov5, yolov8, yolov9, yolov10, yolov11 중 하나이며 seg, pose, obb가 아닌 경우에만 track_box와 track_label을 보여줌
        if ("yolov5" in model_name or
            "yolov8" in model_name or
            "yolov9" in model_name or
            "yolov10" in model_name or
            "yolo11" in model_name) and \
                not self.checkSegName(model_name) and \
                not self.checkPoseName(model_name) and \
                not self.checkObbName(model_name):
            self.ui.track_box.show()
            self.ui.track_label.show()

            return True
        else:
            self.ui.track_box.hide()
            self.ui.track_label.hide()

            return False
    def loadCategories(self, yolo_thread, mode):
        # 클래스 이름 가져오기
        model = None
        class_names = None
        file_name = os.path.basename(yolo_thread.new_model_name)

        if 'yolo' in file_name:
            model = YOLO(yolo_thread.new_model_name)
        elif 'rtdetr' in file_name:
            model = RTDETR(yolo_thread.new_model_name)
        elif 'fastsam' in file_name.lower():
            model = FastSAM(yolo_thread.new_model_name)
        else:
            self.ui.category_box.clearCategories()
            self.ui.category_box.reset_display_text()
            return

        class_names = model.names

        if mode == 'single':
            self.ui.category_box.clearCategories()
            self.ui.category_box.reset_display_text()
            self.ui.category_box.addCategories(class_names)
        elif mode == 'left':
            self.ui.category_box1.clearCategories()
            self.ui.category_box1.reset_display_text()
            self.ui.category_box1.addCategories(class_names)
        elif mode == 'right':
            self.ui.category_box2.clearCategories()
            self.ui.category_box2.reset_display_text()
            self.ui.category_box2.addCategories(class_names)

    def updateCategories(self,yolo_thread, mode):
        # 클래스 이름 가져오기
        if mode == 'single':
            yolo_thread.categories = self.ui.category_box.get_selected_categories()
        elif mode == 'left':
            yolo_thread.categories = self.ui.category_box1.get_selected_categories()
        elif mode == 'right':
            yolo_thread.categories = self.ui.category_box2.get_selected_categories()

    # 내보내기 결과 상태(탐지된 결과)
    def saveStatus(self):
        if self.ui.save_status_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run image results are not saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_res = False
            self.ui.save_button.setEnabled(False)
        elif self.ui.save_status_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run image results will be saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_res = True
            self.ui.save_button.setEnabled(True)

    # 내보내기 결과 상태(탐지된 라벨값 위치 기록)
    def saveLabel(self):
        if self.ui.save_label_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run label results are not saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_label = False
        elif self.ui.save_label_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run label results will be saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_label = True

    # 내보내기 결과 상태(탐지된 이미지 정보 기록)
    def saveJson(self):
        if self.ui.save_json_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run json results are not saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_Json = False
        elif self.ui.save_json_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run json results will be saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_Json = True

    # 테스트 결과 내보내기 --- 프로세스 코드
    def saveResultProcess(self, outdir, current_model_name, folder):
        yolo_thread = self.yolo_threads.get(current_model_name)
        if folder:
            try:
                output_dir = os.path.dirname(yolo_thread.res_path)
                if not os.path.exists(output_dir):
                    self.showStatus('Please wait for the result to be generated')
                    return
                for filename in os.listdir(output_dir):
                    source_path = os.path.join(output_dir, filename)
                    destination_path = os.path.join(outdir, filename)
                    if os.path.isfile(source_path):
                        shutil.copy(source_path, destination_path)
                self.showStatus('Saved Successfully in {}'.format(outdir))
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")
        else:
            try:
                if not os.path.exists(yolo_thread.res_path):
                    self.showStatus('Please wait for the result to be generated')
                    return
                shutil.copy(yolo_thread.res_path, outdir)
                self.showStatus('Saved Successfully in {}'.format(outdir))
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")

    def loadAndSetParams(self, config_file, params):
        if not os.path.exists(config_file):
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                params.update(json.load(f))
        return params

    # 로드 설정 표시줄
    def loadConfig(self):
        params = {"iou": round(random.uniform(0, 1), 2),
                  "conf": round(random.uniform(0, 1), 2),
                  "delay": random.randint(10, 50),
                  "line_thickness": random.randint(1, 5)}
        self.updateParams(params)

        params = {"iou": 0.45, "conf": 0.25, "delay": 10, "line_thickness": 3}
        params = self.loadAndSetParams('config/setting.json', params)
        self.updateParams(params)

    # Config 모델 파라미터 업데이트
    def updateParams(self, params):
        self.ui.iou_spinbox.setValue(params['iou'])
        self.ui.iou_slider.setValue(int(params['iou'] * 100))
        self.ui.conf_spinbox.setValue(params['conf'])
        self.ui.conf_slider.setValue(int(params['conf'] * 100))
        self.ui.speed_spinbox.setValue(params['delay'])
        self.ui.speed_slider.setValue(params['delay'])
        self.ui.line_spinbox.setValue(params['line_thickness'])
        self.ui.line_slider.setValue(params['line_thickness'])

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

    # 하이퍼파라미터 튜닝
    def changeValue(self, x, flag):
        if flag == 'iou_spinbox':
            self.ui.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.ui.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.showStatus('IOU Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.ui.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.ui.conf_spinbox.setValue(x / 100)
            self.showStatus('Conf Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.ui.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.ui.speed_spinbox.setValue(x)
            self.showStatus('Delay: %s ms' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.speed_thres = x  # ms
        elif flag == 'line_spinbox':
            self.ui.line_slider.setValue(x)
        elif flag == 'line_slider':
            self.ui.line_spinbox.setValue(x)
            self.showStatus('Line Width: %s' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.line_thickness = x

    # YOLOv5 및 YOLOv9를 수정하여 yolo.py 충돌 해결
    def solveYoloConflict(self, ptnamelst):
        glo.set_value("yoloname", "yolov5 yolov8 yolov9 yolov9-seg yolov5-seg yolov8-seg rtdetr yolov8-pose yolo11 yolo11-seg yolo11-pose")
        self.reloadModel()

    # 통계 결과를 수락하고 json에 기록
    def setResultStatistic(self, value):
        # --- 통계 결과 얻기 + 히스토그램 그리기 --- #
        self.result_statistic = value
        self.statistic_plot = PlotWindow(self.current_workpath)
        self.statistic_plot.startResultStatistic(self.result_statistic)
        # --- 통계 결과 얻기 + 히스토그램 그리기 --- #

    # 막대 차트 결과 표시
    def showResultStatics(self):
        self.resutl_statistic = dict()
        # JSON 파일 읽기
        with open(self.current_workpath + r'/config/result.json', 'r', encoding='utf-8') as file:
            self.result_statistic = json.load(file)

        if self.result_statistic:
            # Display chart in a new window
            self.chart_view = ResultChartView(self.result_statistic, self)
            self.chart_view.show()

        else:
            print("No completed target detection results detected, please execute the detection task first!")

    # 테이블 결과 목록 가져오기
    def setTableResult(self, results_table, results_error):
        self.detect_result = results_table
        self.detect_errors = results_error

    # 테이블 결과 표시
    def showTableResult(self):
        self.table_result = TableViewQWidget(infoList=self.detect_result, errorList=self.detect_errors)
        self.table_result.show()

    # 테이블 결과 종료
    def closeTableResult(self):
        '''
            중요!!!

            결과 테이블 UI가 Visible 되어 있는 상태에서 Run 버튼을 여러번 클릭 할경우 UI 쓰레드랑 충돌
            나는 문제로 죽는 현상 발생됨, Detected 시작/종료 될때 결과 테이블이 Visible이면 종료 되도록
            해야함
        '''
        if isinstance(self.table_result, TableViewQWidget):
            if self.table_result.isVisible():
                self.table_result.close()

    def modelnamethreshold(self, model_label, model_name, mode='single'):
        thresholdTextSize = 170 if mode == 'single' else 125

        # 원본 텍스트에서 ".pt" 제거
        clean_model_name = str(model_name).replace(".pt", "")
        model_label.setText(clean_model_name)

        # 텍스트 길이를 제한하고 필요시 "..." 추가
        elided_text = model_label.fontMetrics().elidedText(clean_model_name, Qt.ElideRight, thresholdTextSize)
        model_label.setText(elided_text)

        # 전체 텍스트를 tooltip으로 설정 (비어있지 않을 경우에만)
        if clean_model_name.strip():  # 텍스트가 비어있지 않은지 확인
            model_label.setToolTip(clean_model_name)
        else:
            model_label.setToolTip("No model name provided")
