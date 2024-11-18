import os

"""
https://github.com/ultralytics/yolov5/issues/11337
"""
def seg_to_bbox(seg_info):
    # 예시 입력: '5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...'
    class_id, *points = seg_info.strip().split()
    points = [float(p) for p in points]
    x_min = min(points[0::2])
    y_min = min(points[1::2])
    x_max = max(points[0::2])
    y_max = max(points[1::2])
    width = x_max - x_min
    height = y_max - y_min
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    if int(class_id) != 0:
        cid = int(class_id)-1
    else:
        cid = int(class_id)

    bbox_info = f"{cid} {x_center} {y_center} {width} {height}"

    return bbox_info

# 레이블 파일이 저장된 디렉토리 경로를 설정하세요
label_dir = '/path/to/your/labels'

# 변환된 레이블을 저장할 디렉토리 경로를 설정하세요 (원본을 덮어쓰지 않도록 주의)
output_dir = 'g:\\@Example\\AI\\@Python_AI\\yolov8\\test\\@datasets\\train_data\\Nude\\total_nude_content_seg\\test - 복사본\\convert_labels'
os.makedirs(output_dir, exist_ok=True)

# 레이블 파일 목록 가져오기
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

for label_file in label_files:
    input_path = os.path.join(label_dir, label_file)
    output_path = os.path.join(output_dir, label_file)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()
        for line in lines:
            # 빈 라인 또는 잘못된 형식의 라인 처리
            if line.strip() == '':
                continue
            try:
                bbox_info = seg_to_bbox(line)
                outfile.write(bbox_info + '\n')
            except Exception as e:
                print(f"파일 {label_file}의 라인 처리 중 오류 발생: {e}")
                continue

    print(f"변환 완료: {label_file}")