import cv2
import os
import numpy as np
import time
from yolocode.YOLOBaseThread import YOLOBaseThread
from PIL import ImageFont, ImageDraw, Image
from paddleocr import PaddleOCR
# from paddleocr.paddleocr import MODEL_URLS
from utils.image_save import ImageSaver
from pathlib import Path
from googletrans import Translator
import asyncio

# https://www.paddleocr.ai/main/en/update/upgrade_notes.html
# https://github.com/PaddlePaddle/PaddleOCR/issues/17539

class PdOCR:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=self.lang,
            enable_mkldnn=False)
        self.img_path = None
        self.ocr_result = {}

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def run_ocr(self, img_path: str):
        self.img_path = img_path
        ocr_text = []
        result = self._ocr.ocr(img_path)

        # PaddleOCR 3.3.x 결과 구조 처리
        if result and len(result) > 0:
            # 3.3.x 버전 형식
            self.ocr_result = result[0]
            if 'rec_texts' in self.ocr_result:
                ocr_text = self.ocr_result['rec_texts']

        if not ocr_text:
            ocr_text = ["No text detected."]

        image, roi_image = self.show_img_with_ocr()

        return ocr_text, image, roi_image

    def show_img_with_ocr(self):
        image = cv2.imread(self.img_path)
        roi_image = image.copy()

        boxes = []
        texts = []

        # PaddleOCR 3.3.x (Dict format)
        # 3.3.x format: 'rec_boxes' is usually [[x1, y1, x2, y2], ...] flat and rec_polys[[x,y], ...] int
        # But sometimes can be different depending on config (cls=True vs return_word_box=True)
        # We normalize everything to 4 points here.
        raw_boxes = self.ocr_result.get('rec_polys', [])
        texts = self.ocr_result.get('rec_texts', [])

        for box in raw_boxes:
            # If box is flatten [xmin, ymin, xmax, ymax]
            if len(box) == 4 and isinstance(box[0], (int, float, np.number)):
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                boxes.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            # If box is points [[x1,y1], [x2,y2]...]
            elif len(box) == 4 and len(box[0]) == 2:
                boxes.append([(int(p[0]), int(p[1])) for p in box])
            else:
                # Fallback or skip
                pass

        # Draw
        for box, text in zip(boxes, texts):
            topLeft = box[0]
            topRight = box[1]
            bottomRight = box[2]
            bottomLeft = box[3]

            cv2.line(roi_image, topLeft, topRight, (255, 117, 0), 2)
            cv2.line(roi_image, topRight, bottomRight, (255, 117, 0), 2)
            cv2.line(roi_image, bottomRight, bottomLeft, (255, 117, 0), 2)
            cv2.line(roi_image, bottomLeft, topLeft, (255, 117, 0), 2)

            # Dynamic font size logic can be inside put_text or calculated here.
            # Currently put_text uses fixed font_size=17 per user's last edit or passed font_size=22 default
            # User's edit passed font_size=17.
            roi_image = self.put_text(roi_image, text, topLeft[0], topLeft[1] - 20, font_size=17)

        return image, roi_image

    def put_text(self, image, text, x, y, color=(124, 0, 213), font_size=22):
        if type(image) == np.ndarray:
            color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(color_coverted)

        font_path = os.path.join(os.getcwd(), 'fonts', 'SourceHanSansSC-VF.ttf')
        image_font = ImageFont.truetype(font_path, font_size)
        draw = ImageDraw.Draw(image)

        # Draw background rectangle
        text_bbox = draw.textbbox((x, y), text, font=image_font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))

        draw.text((x, y), text, font=image_font, fill=color)

        numpy_image = np.array(image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        return opencv_image

class PaddleOCRThread(YOLOBaseThread):
    def __init__(self):
        super(PaddleOCRThread, self).__init__()
        self.task = 'paddle-ocr'
        self.project = 'runs/paddle-ocr'
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None

    def postprocess(self, lang, img, orig_imgs):

        set_lang = lang if lang != '' else 'korean'
        ocr = PdOCR(lang=set_lang)
        source = self.source if isinstance(self.source, list) else [self.source]
        translator = None
        if self.save_res and self.save_path:
            translator = Translator()

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
                self.save_labels(self.save_path, image_file, ocr_text, translator)


    def save_bbox_preds(self, save_path, image_file, result_image):
        image_name = os.path.basename(image_file)
        image_saver = ImageSaver(result_image)
        image_saver.save_image(save_path / image_name)

    def save_labels(self, save_path, image_file, ocr_text, translator):
        image_name = os.path.basename(image_file)
        text_label_filename = Path(os.path.basename(image_name)).stem + '.txt'
        save_text_path = os.path.join(save_path,text_label_filename)
        translation_results = asyncio.run(self.lang_translate(translator, ocr_text))
        with open(save_text_path, 'w', encoding='utf-8') as txtfile:
            for text in ocr_text:
                txtfile.write(text)
                txtfile.write('\n')

            txtfile.write('==================내용 번역 결과==================\n')

            for translate in translation_results:
                txtfile.write(translate)
                txtfile.write('\n')

    async def lang_translate(self, translator, translation_target):
        translation_result = []
        for value in translation_target:
            translation = await translator.translate(value, dest='ko')
            translation_result.append(translation.text)

        return translation_result

if __name__ == '__main__':
    r = PaddleOCRThread()
    r.postprocess(None,None,None)