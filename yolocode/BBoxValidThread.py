from yolocode.YOLOv8Thread import YOLOv8Thread
from pathlib import Path
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
        # self.source 경로 기준으로 data.yaml 위치 확인
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

    def load_labels(self, label_file):
        """YOLO 텍스트 파일에서 라벨 읽기"""
        labels = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        return labels

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
        v = self.save_path
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
            index = index + 1
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

            labels = self.load_labels(label_file)

            # 원본 이미지 전송
            self.send_input.emit(image)

            # BBox 그리기
            result_image = self.draw_bboxes(image.copy(), labels)

            # 결과 이미지 전송
            self.send_output.emit(result_image)

            # 상태 메시지 전송
            self.send_msg.emit("bbox validation: ({} / {}) {}".format(index, total_count, image_file))

            percent = (index / total_count) * 100 if total_count > 0 else 0
            self.send_progress.emit(percent)