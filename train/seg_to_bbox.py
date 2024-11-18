import os
import numpy as np
from ultralytics.utils.ops import segments2boxes

def convert_label_file(label_path, output_path):
    """
    하나의 레이블 파일을 변환합니다.
    Args:
        label_path (str): 원본 세그멘테이션 레이블 파일 경로
        output_path (str): 변환된 바운딩 박스 레이블 파일을 저장할 경로
    """
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

    with open(output_path, 'w') as f:
        for label in converted_labels:
            f.write(label + '\n')

def convert_labels_in_directory(label_dir, output_dir):
    """
    디렉토리 내의 모든 레이블 파일을 변환합니다.
    Args:
        label_dir (str): 원본 레이블 파일들이 있는 디렉토리 경로
        output_dir (str): 변환된 레이블 파일을 저장할 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        convert_label_file(label_path, output_path)
        print(f"변환 완료: {label_file}")

# 사용 예시
label_dir = 'g:\\Finalmobile6_git\\seg To bbbox\\valid\\labels'  # 원본 세그멘테이션 레이블 디렉토리 경로
output_dir = 'g:\\Finalmobile6_git\\seg To bbbox\\valid\\convert_labels'       # 변환된 바운딩 박스 레이블 디렉토리 경로

convert_labels_in_directory(label_dir, output_dir)
