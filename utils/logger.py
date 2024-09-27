import logging
from colorlog import ColoredFormatter

class LoggerUtils:
    def __init__(self):
        # 로거 생성
        self.logger = logging.getLogger('example_logger')
        self.logger.setLevel(logging.INFO)

        # 반복적으로 추가하는 것을 방지하려면 로거에 이미 프로세서가 있는지 확인하세요.
        if not self.logger.hasHandlers():
            # 컬러 로그 형식 만들기
            log_format = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
            formatter = ColoredFormatter(log_format)

            # 스트림 프로세서를 생성하여 로거에 추가
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(str(msg))


if __name__ == "__main__":
    loggertool = LoggerUtils()
    loggertool.logger.info("test")