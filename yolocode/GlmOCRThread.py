import cv2
from yolocode.YOLOBaseThread import YOLOBaseThread
from pathlib import Path
from glmocr.api import GlmOcr

class GlmOCRThread(YOLOBaseThread):
    def __init__(self):
        super(GlmOCRThread, self).__init__()
        self.task = 'glm-ocr'
        self.project = 'runs/glm-ocr'
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None

    def postprocess(self, lang, img, orig_imgs):
        source = self.source if isinstance(self.source, list) else [self.source]
        with GlmOcr() as parser:
            for image_file in source:
                image = cv2.imread(image_file)
                # 원본 이미지 전송
                self.send_input.emit(image)
                # 결과 이미지 전송
                #self.send_output.emit(roi_image)
                try:
                    result = parser.parse(str(image_file))
                    v = result
                except Exception as e:
                    #print(f"Failed: {p.name}: {e}")
                    continue
