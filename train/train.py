from ultralytics import YOLO

# Load a model
model = YOLO("c:\\Users\\USER\\Downloads\\total_nude_content\\yolov10x.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="c:\\Users\\USER\\Downloads\\total_nude_content\\data.yaml", batch=3, epochs=300, imgsz=640, device=[0])