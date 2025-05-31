from ultralytics import YOLO
from multiprocessing import freeze_support
import os

if __name__ == '__main__':
    # freeze_support()  # 멀티프로세싱 지원을 위해 필요

    # 체크포인트 파일 경로 확인
    checkpoint_path = "e:\\@Example\\AI\\@Python_AI\\yolov8\\yolov8_GUI\\YoloView\\runs\\detect\\train\\weights\\last.pt"  # 또는 실제 저장된 경로

    # 체크포인트 파일이 존재하는 경우 재개, 없으면 새로 시작
    if os.path.exists(checkpoint_path):
        print(f"체크포인트 발견: {checkpoint_path}에서 학습 재개")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
    else:
        print("새로운 학습 시작")
        # Load a model
        model = YOLO("e:\\@Example\\AI\\@Python_AI\\yolov8\\yolov8_GUI\\YoloView\\ptfiles\\yolo11m.pt")

        # Train the model with GPU
        results = model.train(data="d:\\total_nude_content_bbox\\data.yaml", epochs=300, imgsz=640, device=[0])