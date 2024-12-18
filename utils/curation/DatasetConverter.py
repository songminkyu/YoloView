import os
import numpy as np
from ultralytics.utils.ops import segments2boxes
from ultralytics.data.converter import yolo_bbox2segment

class DatasetConverter:
    def __init__(self, root_folder, subfolders):
        self.root_folder = root_folder
        self.subfolders = subfolders # ['train','valid','test']

    def convert_label_segments_to_bboxes(self,label_path):

        with open(label_path, 'r') as f:
            lines = f.readlines()

        converted_labels = []
        for line in lines:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 == 0 and len(coords) >= 6:
                # 세그먼트(폴리곤)인 경우
                segment = np.array(coords).reshape(-1, 2)
                segments = [segment]
                boxes = segments2boxes(segments)
                for box in boxes:
                    x_center, y_center, width, height = box
                    converted_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
            else:
                # 좌표가 부족한 경우 (처리할 수 없는 라인)
                print(f"파일 {label_path}의 라인을 건너뜁니다: {line.strip()}")
                continue

        with open(label_path, 'w') as f:
            for label in converted_labels:
                f.write(label + '\n')

    def segments_to_boxes(self):

        for subfolder in self.subfolders:
            label_dir = os.path.join(self.root_folder, subfolder, "labels")
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            for label_file in label_files:
                label_path = os.path.join(label_dir, label_file)
                self.convert_label_segments_to_bboxes(label_path)


# 실행 예시
if __name__ == "__main__":
   root_folder = "c:\\Users\\USER\\Downloads\\FAZ FINAL.v2i.yolov11"
   subfolders = ["train", "test", "valid"]
   d = DatasetConverter(root_folder,subfolders)
   d.segments_to_boxes()


