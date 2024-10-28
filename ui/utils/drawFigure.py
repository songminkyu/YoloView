import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

'''
중요 : matplotlib은 기본 대화형 GUI 백엔드인데 matplotlib.use('agg') 선언 해줌으로써 비대화형 백엔드로 전환되어, 메인쓰레드와
UI전용 워커쓰레드간에 충돌을 피할수 있게 해주는 기능
해줌. 
1. matplotlib.use('agg') (AGG 백엔드)
    AGG (Anti-Grain Geometry): 렌더링된 이미지를 파일에 저장하는 데 최적화된 백엔드입니다.
    특징:
        GUI와 상관없이 사용할 수 있는 비대화형 백엔드입니다.
        주로 PNG, PDF, SVG 등 이미지 파일로 출력하기 위해 사용됩니다.
        디스플레이 장치가 없는 환경(예: 서버, 배치 작업)에서 주로 사용됩니다.
        속도가 빠르고 높은 품질의 이미지를 생성합니다.
'''

class PlottingThread(QThread):
    # Signal to send processed data to the main GUI thread
    plot_data_ready = Signal(dict)

    def __init__(self, result_statistic):
        super().__init__()
        self.result_statistic = result_statistic

    def run(self):
        # Calculate percentages in the background thread
        total = sum(self.result_statistic.values())
        percentages = {k: (v / total * 100) for k, v in self.result_statistic.items()}

        # Emit signal with calculated percentages
        self.plot_data_ready.emit(percentages)


class PlotWindow(QWidget):
    def __init__(self, workpath):
        super().__init__()
        self.workpath = workpath
        self.result_statistic = {}
        # Placeholder for the PlottingThread
        self.plot_thread = None

    def startResultStatistic(self, value):
        # Write JSON data to file
        with open('config/result.json', 'w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=4)

        # Update the result statistics and start the thread
        self.result_statistic = value
        self.startPlotThread()

    def startPlotThread(self):
        if self.plot_thread is not None:
            # Clean up any previous thread if it's running
            self.plot_thread.quit()
            self.plot_thread.wait()

        # Create and start a new PlottingThread
        self.plot_thread = PlottingThread(self.result_statistic)
        self.plot_thread.plot_data_ready.connect(self.plot_data)  # Connect the signal to plot_data method
        self.plot_thread.start()

    def plot_data(self, percentages):
        # Plotting handled in the main thread
        activities = list(percentages.keys())
        values = list(percentages.values())

        plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

        plt.figure(figsize=(10, 6))
        bars = plt.bar(activities, values, color='skyblue')

        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')

        # Add title and labels
        plt.title('Detection results target category statistical proportion')
        plt.xlabel('Target Category')
        plt.ylabel('Percentage (%)')

        # Save the plot to a file
        plt.savefig(self.workpath + r'\config\result.png')
        plt.close()
