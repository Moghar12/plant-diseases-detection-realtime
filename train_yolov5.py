from ultralytics import YOLO

model = YOLO('yolov5s.yaml')

model.train(data='dataset/data.yaml', epochs=100, imgsz=640)
