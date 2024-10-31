from utils import glo
import json
import os
import cv2
from PySide6.QtGui import QMouseEvent, QGuiApplication
from PySide6.QtCore import Qt, QPropertyAnimation, Signal
from ui.utils.customGrips import CustomGrip
from yoloshow.YOLOSHOW import YOLOSHOW
from yoloshow.YOLOSHOWVS import YOLOSHOWVS


class YOLOSHOWWindow(YOLOSHOW):
    # 종료 신호 정의
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWWindow, self).__init__()
        self.center()
        # --- 창을 드래그하여 창 크기를 변경 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True)  # ==> 설정 창은 드래그를 지원합니다(설정해야 함).
        # --- 창을 드래그하여 창 크기를 변경하세요 --- #
        self.animation_window = None
        self.drag = None

        # 마우스 드래그 이벤트
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 파일인지 확인
            event.acceptProposedAction()  # 드래그 앤 드롭 데이터 허용


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 모든 파일 경로 가져옴
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 파일 경로 가져오기
        if file:
            # 폴더인지 확인
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 폴더의 첫 번째 사진 표시
                self.showStatus('Loaded Folder：{}'.format(os.path.basename(file)))
            #사진/비디오
            else:
                self.inputPath = file
                # 동영상인 경우 첫 번째 프레임을 표시.
                if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                    # 첫 번째 프레임을 표시
                    self.cap = cv2.VideoCapture(self.inputPath)
                    ret, frame = self.cap.read()
                    if ret:
                        # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.showImg(frame, self.main_leftbox, 'img')
                # 사진이라면 정상적으로 표시
                else:
                    self.showImg(self.inputPath, self.main_leftbox, 'path')
                self.showStatus('Loaded File：{}'.format(os.path.basename(self.inputPath)))
        glo.set_value('inputPath', self.inputPath)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6 화면의 매개변수를 가져옴.
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)

    # 창을 드래그하여 창 크기를 변경하세요.
    def resizeEvent(self, event):
        # Update Size Grips
        self.resizeGrip()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 여기에 정의된 애니메이션 표시
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 애니메이션 시간 500밀리초
            self.animation.setStartValue(0)  # 완전한 투명성으로 시작
            self.animation.setEndValue(1)  # 완전 불투명으로 종료
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()

# MouseLabel 메소드를 구현하기 위한 여러 클래스 세트
class YOLOSHOWVSWindow(YOLOSHOWVS):
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWVSWindow, self).__init__()
        self.center()
        # --- 창을 드래그하여 창 크기를 변경하세요 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True) # ==> 설정 창은 드래그를 지원합니다(설정해야 함).
        # --- 창을 드래그하여 창 크기를 변경하세요 --- #
        self.animation_window = None


    # 마우스 드래그 이벤트
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 파일인지 확인
            event.acceptProposedAction()  # 드래그 앤 드롭 데이터 허용


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 모든 파일 경로 가져오기
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 파일 경로 가져오기
        if file:
            # 폴더인지 확인
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 폴더의 첫 번째 사진 표시
                self.showStatus('Loaded Folder：{}'.format(os.path.basename(file)))
            # 사진/비디오
            else:
                self.inputPath = file
                # 비디오인 경우 첫 번째 프레임을 표시.
                if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                    # 첫 번째 프레임 표시
                    self.cap = cv2.VideoCapture(self.inputPath)
                    ret, frame = self.cap.read()
                    if ret:
                        # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.showImg(frame, self.main_leftbox, 'img')
                # 사진이라면 정상적으로 표시 됨.
                else:
                    self.showImg(self.inputPath, self.main_leftbox, 'path')
                self.showStatus('Loaded File：{}'.format(os.path.basename(self.inputPath)))
        glo.set_value('inputPath', self.inputPath)


    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6은 화면 매개변수를 얻습니다.
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)

    # 창을 드래그하여 창 크기를 변경
    def resizeEvent(self, event):
        # Update Size Grips
        self.resizeGrip()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 여기에 정의된 애니메이션 표시
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 애니메이션 시간 500밀리초
            self.animation.setStartValue(0)  # 완전한 투명성으로 시작
            self.animation.setEndValue(1)  # 완전 불투명으로 끝남
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()