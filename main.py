import sys
import os
import logging
# 시스템 경로에 ui 디렉터리를 추가
sys.path.append(os.path.join(os.getcwd(), "ui"))
# 표준 출력 비활성화
sys.stdout = open(os.devnull, 'w')
logging.disable(logging.CRITICAL)  # 모든 수준의 로깅 비활성화
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from utils import glo
from yoloshow.Window import YOLOSHOWWindow as yoloshowWindow
from yoloshow.Window import YOLOSHOWVSWindow as yoloshowVSWindow
from yoloshow.ChangeWindow import yoloshow2vs, vs2yoloshow

if __name__ == '__main__':
    app = QApplication([])  # 애플리케이션 인스턴스 생성
    app.setWindowIcon(QIcon('images/swimmingliu.ico'))  # 애플리케이션 아이콘 설정

    # 모든 QFrame의 테두리를 제거하기 위해 전체 애플리케이션에 대한 스타일시트를 설정
    app.setStyleSheet("QFrame { border: none; }")

    # Window 인스턴스 생성
    yoloshow = yoloshowWindow()
    yoloshowvs = yoloshowVSWindow()

    # 전역 변수 관리자를 초기화하고 값을 설정
    glo._init()  # 初始化全局变量空间
    glo.set_value('yoloshow', yoloshow)  # Yoloshow Window 인스턴스 저장
    glo.set_value('yoloshowvs', yoloshowvs)  # yoloshowvs Window 인스턴스 저장

    # 전역 변수 관리자에서 창 인스턴스 가져오기
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')

    # yoloshow Window 표시
    yoloshow_glo.show()

    # 신호와 슬롯을 연결하여 인터페이스 간 전환
    yoloshow_glo.ui.src_vsmode.clicked.connect(yoloshow2vs)  # 단일 모드에서 대비 모드로 전환
    yoloshowvs_glo.ui.src_singlemode.clicked.connect(vs2yoloshow)  # 대비 모드에서 다시 단일 모드로 전환

    app.exec()  # 애플리케이션의 이벤트 루프 시작
