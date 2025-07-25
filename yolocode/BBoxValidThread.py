import os.path
from yolocode.YOLOv8Thread import YOLOv8Thread
from pathlib import Path
from utils.image_save import ImageSaver
import cv2
import yaml

class BBoxValidThread(YOLOv8Thread):
    def __init__(self):
        super(BBoxValidThread, self).__init__()
        self.task = 'bbox_valid'
        self.project = 'runs/bbox_valid'
        self.data_yaml = 'data.yaml'  # data.yaml 파일 이름만 지정
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None
        self.classes = []

    def load_classes(self):
        """data.yaml에서 클래스 이름 로드"""
        source = self.source[0]
        data_yaml_path = Path(source) / self.data_yaml
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

    def load_bbox_labels(self, label_file):
        """YOLO 텍스트 파일에서 bbox 라벨만 읽기"""
        bbox_labels = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLO bbox 형식: class x_center y_center width height
                    try:
                        bbox_labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        self.send_msg.emit(f"Invalid bbox format in {label_file}: {line.strip()}")
        return bbox_labels

    def draw_bboxes(self, image, labels):
        """Bounding Box를 이미지에 그리기"""
        h, w = image.shape[:2]
        for label in labels:
            class_id, x_center, y_center, box_width, box_height = label
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, self.classes[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def postprocess(self, preds, img, orig_imgs):
        """메인 스레드 실행"""
        self.load_classes()  # 클래스 이름 로드
        for subfolder in ['train', 'valid', 'test']:
            # images 및 labels 경로 설정
            source = os.path.join(self.source[0], subfolder)

            images_path = Path(source) / 'images'
            labels_path = Path(source) / 'labels'

            if not images_path.exists() and not labels_path.exists():
                continue

            # images 폴더에서 이미지 파일 로드
            image_files = []
            for ext in ['*.jpg', '*.png']:
                image_files.extend(list(images_path.glob(ext)))

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

                # YOLO bbox 형식의 라벨만 로드
                labels = self.load_bbox_labels(label_file)

                # bbox 라벨이 없으면 건너뛰기
                if not labels:
                    self.send_msg.emit(f"No valid bbox labels found in {label_file}")
                    percent = (index / total_count) * 100 if total_count > 0 else 0
                    self.send_progress.emit(percent)
                    continue

                # 원본 이미지 전송
                self.send_input.emit(image)

                # BBox 그리기
                result_image = self.draw_bboxes(image.copy(), labels)

                # 결과 이미지 전송
                self.send_output.emit(result_image)

                # 상태 메시지 전송
                self.send_msg.emit(f"bbox validation: ({index} / {total_count}) {image_file}")

                percent = (index / total_count) * 100 if total_count > 0 else 0
                self.send_progress.emit(percent)

                # 이미지 저장
                if self.save_res and self.save_path:
                    self.save_bbox_preds(self.save_path, image_file, result_image)

    def save_bbox_preds(self, save_path, image_file, result_image):
        image_name = os.path.basename(image_file)
        image_saver = ImageSaver(result_image)
        image_saver.save_image(save_path / image_name)
