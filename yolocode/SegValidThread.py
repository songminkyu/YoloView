import os.path
from yolocode.YOLOv8Thread import YOLOv8Thread
from pathlib import Path
from utils.image_save import ImageSaver
import cv2
import yaml
import numpy as np

class SegValidThread(YOLOv8Thread):
    def __init__(self):
        super(SegValidThread, self).__init__()
        self.task = 'seg_valid'
        self.project = 'runs/seg_valid'
        self.data_yaml = 'data.yaml'  # data.yaml 파일 이름만 지정
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None
        self.classes = []

    def load_classes(self):
        """data.yaml에서 클래스 이름 로드"""
        source = self.source[0]
        data_yaml_path = Path(source).parent / self.data_yaml
        if not data_yaml_path.exists():
            self.send_msg.emit(f"data.yaml not found at {data_yaml_path}")
            return

        try:
            with open(data_yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                self.classes = data.get('names', [])
                self.send_msg.emit(f"Loaded classes: {self.classes}")
        except Exception as e:
            self.send_msg.emit(f"Failed to load data.yaml: {str(e)}")

    def load_segmentation_labels(self, label_file):
        """YOLO 텍스트 파일에서 Segmentation 라벨 읽기"""
        seg_labels = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 5:  # 클래스 ID와 좌표가 있어야 함
                    class_id = int(parts[0])  # 클래스 ID
                    points = list(map(float, parts[1:]))  # 폴리곤 좌표
                    if len(points) % 2 == 0:  # (x, y) 쌍으로 나뉘어야 함
                        seg_labels.append({
                            'class_id': class_id,
                            'polygon': np.array(points, dtype=np.float32).reshape(-1, 2)  # (x, y) 형태로 변환
                        })
        return seg_labels

    def draw_segmentation_masks(self, image, seg_labels):
        """Segmentation 마스크와 라벨을 이미지에 그리기"""
        h, w = image.shape[:2]
        overlay = image.copy()

        for seg_label in seg_labels:
            class_id = seg_label['class_id']
            polygon = seg_label['polygon']

            # 좌표를 이미지 크기에 맞게 스케일링
            scaled_polygon = (polygon * [w, h]).astype(np.int32)

            # 랜덤 색상 생성
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            # 마스크를 이미지에 적용
            cv2.fillPoly(overlay, [scaled_polygon], color)

            # 좌측 상단 좌표 계산 (x, y 값이 모두 가장 작은 점)
            top_left_point = scaled_polygon[np.lexsort((scaled_polygon[:, 1], scaled_polygon[:, 0]))][0]

            # 라벨 텍스트
            label_text = self.classes[class_id] if self.classes else str(class_id)

            # 라벨 텍스트 추가
            cv2.putText(overlay, label_text, (top_left_point[0], top_left_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 마스크와 원본 이미지를 병합
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return blended

    def postprocess(self, preds, img, orig_imgs):
        """메인 스레드 실행"""
        self.load_classes()  # 클래스 이름 로드

        # images 및 labels 경로 설정
        source = self.source[0]
        images_path = Path(source) / 'images'
        labels_path = Path(source) / 'labels'

        # images 폴더에서 JPG 파일 로드
        image_files = list(images_path.glob('*.jpg'))
        if not image_files:
            self.send_msg.emit(f"No images found in {images_path}")
            return

        percent = 0
        index = 0
        total_count = len(image_files)
        for image_file in image_files:
            index += 1

            # labels 폴더에서 라벨 파일 경로 생성
            label_file = labels_path / f"{image_file.stem}.txt"
            if not label_file.exists():
                self.send_msg.emit(f"No label file found for {image_file}")
                continue

            # 이미지 및 라벨 읽기
            image = cv2.imread(str(image_file))
            if image is None:
                self.send_msg.emit(f"Failed to read image: {image_file}")
                continue

            # YOLO segment 형식의 라벨만 로드
            labels = self.load_segmentation_labels(label_file)

            # segment 라벨이 없으면 건너뛰기
            if not labels:
                self.send_msg.emit(f"No valid seg labels found in {label_file}")
                percent = (index / total_count) * 100 if total_count > 0 else 0
                self.send_progress.emit(percent)
                continue

            # 원본 이미지 전송
            self.send_input.emit(image)

            # segment 그리기
            result_image = self.draw_segmentation_masks(image.copy(), labels)

            # 결과 이미지 전송
            self.send_output.emit(result_image)

            # 상태 메시지 전송
            self.send_msg.emit(f"seg validation: ({index} / {total_count}) {image_file}")

            percent = (index / total_count) * 100 if total_count > 0 else 0
            self.send_progress.emit(percent)

            # 이미지 저장
            if self.save_res and self.save_path:
                self.save_seg_preds(self.save_path, image_file, result_image)

    def save_seg_preds(self, save_path, image_file, result_image):
        image_name = os.path.basename(image_file)
        image_saver = ImageSaver(result_image)
        image_saver.save_image(save_path / image_name)
