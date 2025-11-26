from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import math
import numpy

"""model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
results = model.train(data='C:/KISISEL/DERS/Projeler/Python/calisma/data/classification', epochs=1, imgsz=64)"""

# model = YOLO("C:/KISISEL/DERS/Projeler/Python/calisma/runs/classify/train/weights/last.pt")  # load a custom model

# results = model("C:/KISISEL/DERS/Projeler/Python/calisma/data/classification/predict/frame_0aaa.png")  # predict on an image

# names = results[0].names
# probs_t = results[0].probs.data        # -> torch.Tensor
# probs_list = probs_t.detach().cpu().numpy().tolist()
# print(probs_list)

# model = YOLO("yolo11n-cls.yaml")
# results = model.train(data='C:/KISISEL/DERS/Projeler/Python/calisma/data/classification', epochs=100, imgsz=64)

# model = YOLO('C:/KISISEL/DERS/Projeler/Python/calisma/runs/classify/train2/weights/best.pt')
# metrics = model.val()
# metrics.top5

model = YOLO('C:/KISISEL/DERS/Projeler/Python/calisma/runs/classify/train2/weights/best.pt')
results = model("C:/KISISEL/DERS/Projeler/Python/calisma/data/classification/predict/")

for result in results:
    # result.plot() fonksiyonu, üzerine yazı/kutu çizilmiş resmi numpy array olarak döner
    plotted_img = result.plot()

    # Yine BGR -> RGB dönüşümü şart
    plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(plotted_img_rgb)
    plt.axis('off')
    plt.show()


# num_images = len("C:/KISISEL/DERS/Projeler/Python/calisma/data/classification/predict/")
# cols = 3
# rows = math.ceil(num_images/cols)
# plt.figure(figsize=(12, 6 * rows))
# for i, result in enumerate(results):
#     img_array = result.orig_img
#     img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#     top1_index = result.probs.top1
#     # top2_index = result.probs.top2
#     # top3_index = result.probs.top3
#     conf_score = result.probs.top1conf.item()
#     class_name = result.names[top1_index]
#     plt.subplot(rows, cols, i + 1)
#     plt.imshow(img_rgb)
#     plt.title(f"{class_name}\n%{conf_score * 100:.1f}",
#               color='darkgreen',
#               fontsize=11,
#               fontweight='bold',
#               pad=10)
#     plt.axis('off')
#
# plt.subplots_adjust(hspace=50, wspace=30)
# plt.tight_layout()
# plt.show()



# names = results[0].names
# probs_t = results[0].probs.data        # -> torch.Tensor
# probs_list = probs_t.detach().cpu().numpy().tolist()
# print(probs_list)

