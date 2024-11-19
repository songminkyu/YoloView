from ultralytics import YOLO

'''
yolov8s.pt 을 onnx로 업그레이드 할때 사용하는 코드. 
'''
model = YOLO("yolov8s.pt")

model.export(format='onnx')