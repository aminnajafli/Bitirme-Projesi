from ultralytics import YOLO
model = YOLO("yolo11n.pt") 
results = model.train(data="data.yaml", epochs=3, imgsz=640)
