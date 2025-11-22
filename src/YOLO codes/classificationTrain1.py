from ultralytics import YOLO


model = YOLO("data.yaml") 
model = YOLO("yolo11n-cls.pt")  
model = YOLO("data.yaml").load("yolo11n-cls.pt")  

results = model.train(data="data.yaml", epochs=100, imgsz=64)
