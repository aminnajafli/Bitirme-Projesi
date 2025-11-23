from ultralytics import YOLO
import numpy

"""model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
results = model.train(data='C:/KISISEL/DERS/Projeler/Python/calisma/data/classification', epochs=1, imgsz=64)"""

model = YOLO("C:/KISISEL/DERS/Projeler/Python/calisma/runs/classify/train/weights/last.pt")  # load a custom model

results = model("C:/KISISEL/DERS/Projeler/Python/calisma/data/classification/predict/frame_0aaa.png")  # predict on an image

names = results[0].names
probs_t = results[0].probs.data        # -> torch.Tensor
probs_list = probs_t.detach().cpu().numpy().tolist()
print(probs_list)



