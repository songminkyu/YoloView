import os
import cv2

# 이미지와 레이블 디렉토리 경로 설정
image_dir = '/path/to/your/images'
label_dir = '/path/to/your/labels'

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

for image_file in image_files:
    # 이미지와 대응되는 레이블 파일 경로 설정
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

    # 이미지 로드
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # 레이블 파일 존재 여부 확인
    if not os.path.exists(label_path):
        print(f"레이블 파일이 없습니다: {label_path}")
        continue

    # 레이블 로드 및 바운딩 박스 그리기
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = label.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 상대 좌표를 절대 좌표로 변환
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height

            # 바운딩 박스 좌표 계산
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            x_max = int(x_center_abs + width_abs / 2)
            y_max = int(y_center_abs + height_abs / 2)

            # 바운딩 박스 그리기
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 클래스 ID 표시 (선택 사항)
            cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 이미지 표시
    cv2.imshow('Labeled Image', img)
    key = cv2.waitKey(0)
    if key == 27:  # Esc 키를 누르면 종료
        break

cv2.destroyAllWindows()