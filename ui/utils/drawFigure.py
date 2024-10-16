import matplotlib.pyplot as plt
from PySide6.QtCore import QThread


class PlottingThread(QThread):
    def __init__(self, result_statistic, workpath):
        super().__init__()
        self.result_statistic = result_statistic
        self.workpath = workpath

    def run(self):
        # 중국어 글꼴 설정
        plt.rcParams['font.sans-serif'] = ['SimHei']  # '심헤이'는 중국에서 흔히 볼 수 있는 볼드체입니다.
        plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 '-'가 사각형으로 표시되는 문제 해결

        # 합계를 계산하다
        total = sum(self.result_statistic.values())
        # 각 범주의 비율 계산
        percentages = {k: (v / total * 100) for k, v in self.result_statistic.items()}

        # 데이터 준비
        activities = list(percentages.keys())
        values = list(percentages.values())

        # 히스토그램 만들기
        plt.figure(figsize=(10, 6))  # 그래픽의 표시 크기 설정
        bars = plt.bar(activities, values, color='skyblue')  # 绘制柱状图

        # 각 막대 위에 백분율 추가
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')

        # 제목 및 태그 추가
        plt.title('Detection results target category statistical proportion')
        plt.xlabel('Target Category')
        plt.ylabel('Percentage (%)')

        # 그래픽을 파일에 저장
        plt.savefig(self.workpath + r'\config\result.png')
        plt.close()  # 중요: 그래픽을 닫아 메모리를 확보하세요
