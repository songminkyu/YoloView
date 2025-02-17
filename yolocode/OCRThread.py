from yolocode.YOLOv8Thread import YOLOv8Thread
import cv2
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.paddleocr import MODEL_URLS
from utils.image_save import ImageSaver
import os

class PdOCR:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True,lang=self.lang)
        self.img_path = None
        self.ocr_result = {}

    def get_available_langs(self):
        langs_info = []

        for idx, model_name in enumerate(list(MODEL_URLS['OCR'].keys())):
            for lang in list(MODEL_URLS['OCR'][model_name]['rec'].keys()):
                if lang not in langs_info:
                    langs_info.append(lang)

        print('Available Language : {}'.format(langs_info))

    def get_available_models(self):
        model_info = {}

        for idx, model_name in enumerate(list(MODEL_URLS['OCR'].keys())):
            model_info[model_name] = list(MODEL_URLS['OCR'][model_name]['rec'].keys())
            print('#{} Model Vesion : [{}] - Language : {}'.format(idx + 1, model_name,
                                                                   list(MODEL_URLS['OCR'][model_name]['rec'].keys())))

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def run_ocr(self, img_path: str):
        self.img_path = img_path
        ocr_text = []
        result = self._ocr.ocr(img_path, cls=True)
        self.ocr_result = result[0]

        if self.ocr_result:
            for r in result[0]:
                ocr_text.append(r[1][0])
        else:
            ocr_text = "No text detected."

        image, roi_image = self.show_img_with_ocr()

        return ocr_text, image, roi_image

    def show_img_with_ocr(self):
        image = cv2.imread(self.img_path)
        roi_image = image.copy()

        for text_result in self.ocr_result:
            text = text_result[1][0]
            tlX = int(text_result[0][0][0])
            tlY = int(text_result[0][0][1])
            trX = int(text_result[0][1][0])
            trY = int(text_result[0][1][1])
            brX = int(text_result[0][2][0])
            brY = int(text_result[0][2][1])
            blX = int(text_result[0][3][0])
            blY = int(text_result[0][3][1])

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            topLeft = pts[0]
            topRight = pts[1]
            bottomRight = pts[2]
            bottomLeft = pts[3]

            cv2.line(roi_image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_image, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_image = self.put_text(roi_image, text, topLeft[0], topLeft[1] - 20, font_size=17)

        return image, roi_image

    def put_text(self, image, text, x, y, color=(0, 255, 0), font_size=22):
        if type(image) == np.ndarray:
            color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(color_coverted)

        font_path=os.path.join(os.getcwd(),'fonts','SourceHanSansSC-VF.ttf')
        image_font = ImageFont.truetype(font_path, font_size)
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(image)

        draw.text((x, y), text, font=image_font, fill=color)

        numpy_image = np.array(image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        return opencv_image

class OCRThread(YOLOv8Thread):
    def __init__(self):
        super(OCRThread, self).__init__()
        self.task = 'ocr'
        self.project = 'runs/orc'
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None

    def postprocess(self, lang, img, orig_imgs):
        ocr = PdOCR(lang=lang)
        source = self.source if isinstance(self.source, list) else [self.source]

        percent = 0
        index = 0
        total_count = len(source)
        for image_file in source:
            index += 1
            ocr_text, image, roi_image = ocr.run_ocr(image_file)
            # 원본 이미지 전송
            self.send_input.emit(image)
            # 결과 이미지 전송
            self.send_output.emit(roi_image)

            # 상태 메시지 전송
            self.send_msg.emit(f"OCR Detecting : ({index} / {total_count}) {image_file}")

            percent = (index / total_count) * 100 if total_count > 0 else 0
            self.send_progress.emit(percent)

            # 이미지 저장
            if self.save_res and self.save_path:
                self.save_bbox_preds(self.save_path, image_file, roi_image)

    def save_bbox_preds(self, save_path, image_file, result_image):
        image_name = os.path.basename(image_file)
        image_saver = ImageSaver(result_image)
        image_saver.save_image(save_path / image_name)

    def save_labels(self, save_path, image_file, result_image):
        pass


if __name__ == '__main__':
    r = OCRThread()
    r.postprocess(None,None,None)
