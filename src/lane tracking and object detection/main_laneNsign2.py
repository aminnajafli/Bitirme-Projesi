import cv2
import torch
import numpy as np
import os
import time
from PIL import Image as PILImage
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# Şerit takibi için gerekli utils (sliding_windows klasörü yaninda olmali)
# Eger hata verirse sys.path.append eklemek gerekebilir.
from sliding_windows.utils import select_device, lane_line_mask, letterbox

# ---------------------------------------------------------
# 1. SINIF: CNN Modeli (Levha Sınıflandırma İçin)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, 43)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

# ---------------------------------------------------------
# 2. SINIF: Levha ve Nesne Tespit Sistemi
# ---------------------------------------------------------
class SignObjectDetector:
    def __init__(self, models_dir):
        print("[INFO] Levha ve Nesne modelleri yükleniyor...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Modelleri yükle
        self.yolo_traffic = YOLO(os.path.join(models_dir, "alltoget_v2.pt"))
        self.yolo_obj = YOLO(os.path.join(models_dir, "object.pt"))
        
        # CNN Modeli
        self.class_model = Net().to(self.device)
        self.class_model.load_state_dict(torch.load(os.path.join(models_dir, "balanced_data_micronnet.pth"), map_location=self.device))
        self.class_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        self.class_names = [
            "20 km/h hiz siniri", "30 km/h hiz siniri", "50 km/h hiz siniri", "60 km/h hiz siniri",
            "70 km/h hiz siniri", "80 km/h hiz siniri", "mecburi sag", "100 km/h hiz siniri",
            "120 km/h hiz siniri", "sollamak yasak", "kamyonlar icin sollamak yasak",
            "ana yol tali yol kavsagi", "park etmek yasak", "yol ver", "dur",
            "tasit trafigine kapali yol", "kamyon giremez", "girisi olmayan yol", "dikkat",
            "sola donus yasak", "saga donus yasak", "sola tehlikeli devamli virajlar",
            "sola mecburi yon", "yol calismasi", "kaygan yol", "donel kavsak", "trafik isareti",
            "yaya geciti", "park", "bisiklet giremez", "gizli buzlanma", "durak",
            "kirmizi isik", "ileriden saga mecburi yon", "ileriden sola mecburi yon", "ileri mecburi yon",
            "ileri ve saga mecburi yon", "ileri ve sola mecburi yon", "sagdan gidiniz", "soldan gidiniz",
            "sari isik", "yesil isik", "sagdan daralan yol"
        ]
        self.obj_labels = ["insan","bisiklet", "motor", "araba", "otobus", "kamyon"]

    def process(self, frame):
        # Frame üzerinde değişiklik yapacağımız için kopyalamaya gerek yok, direkt üstüne çizebiliriz.
        # Ancak PIL dönüşümü için RGB'ye çevirmeliyiz.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. Trafik Levhaları ---
        try:
            results = self.yolo_traffic(frame_rgb, verbose=False) # verbose=False terminali temiz tutar
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    # Sınır kontrolü (crop işlemi hatasını önlemek için)
                    h, w, _ = frame.shape
                    b[0] = max(0, b[0]); b[1] = max(0, b[1])
                    b[2] = min(w, b[2]); b[3] = min(h, b[3])

                    cropped = frame_rgb[b[1]:b[3], b[0]:b[2]]
                    if cropped.size == 0: continue

                    img_pil = PILImage.fromarray(cropped)
                    img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.class_model(img_tensor)
                        _, pred = torch.max(output, 1)
                        label = self.class_names[pred.item()]

                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            print(f"Levha hatası: {e}")

        # --- 2. İnsan/Araç Tespiti ---
        try:
            # 1. GÜVEN EŞİĞİ EKLEME (conf=0.50)
            results_obj = self.yolo_obj(frame_rgb, verbose=False, conf=0.50, iou=0.45)
            
            for r in results_obj:
                for box in r.boxes:
                    cls_id = int(box.cls.cpu().numpy())
                    if cls_id >= len(self.obj_labels): continue
                    
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 2. SINIF BİRLEŞTİRME MANTIĞI
                    raw_label = self.obj_labels[cls_id]
                    if raw_label in ["otobus", "kamyon"]:
                        label = "AGIR VASITA"
                    else:
                        label = raw_label

                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    # ... (kalan kod aynı)

                    cv2.putText(frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    if label == "insan": 
                        cv2.putText(frame, "DIKKAT! INSAN!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        except Exception as e:
            print(f"Nesne hatası: {e}")

        return frame

# ---------------------------------------------------------
# 3. SINIF: Şerit Takip Sistemi
# ---------------------------------------------------------
class LaneDetector:
    def __init__(self, model_path):
        print("[INFO] Şerit modeli yükleniyor...")
        self.device = select_device('0')
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        self.half = True
        if self.half:
            self.model.half()
        
        # Takip hafızası (titremeyi önlemek için)
        self.lanes = [[0, 0], [0, 0]]
        self.close_lanes = [999, 999]

    def process(self, frame):
        try:
            # YOLO girişi için ön işleme (letterbox)
            img = letterbox(frame, 640, stride=32)[0]
            if img.shape[2] == 4: img = img[:, :, :3]
            
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3: img = img.unsqueeze(0)

            # Model Tahmini
            _, _, ll = self.model(img)
            ll_seg_mask = lane_line_mask(ll)
            ll = ll.squeeze().detach().cpu().numpy().astype(np.uint8)

            # Maskeyi orijinal boyuta getir
            if ll.shape != frame.shape[:2]:
                ll = cv2.resize(ll, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # --- Şerit Çizimi (Kırmızı) ---
            foreground = ll == 1
            frame[foreground] = [0, 0, 255] # Şeritleri kırmızıya boya

            # --- Orta Nokta Hesabı ---
            contours, _ = cv2.findContours((ll_seg_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_width = ll_seg_mask.shape[1]
            height, width = img.shape[2:]
            mean = mask_width / 2

            for contour in contours:
                if cv2.contourArea(contour) < 200: continue
                if len(contour) < 20: continue
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Duba kareye yakındır (Oran 0.6-1.6 arası) ve çok devasa değildir.
                if 0.6 < aspect_ratio < 1.6 and cv2.contourArea(contour) < 1200:
                    continue
                
                # Şerit dikeyde uzundur, çok kısaysa (leke ise) atla.
                if h < 30:
                    continue
                
                points = contour[:, 0, :]
                min_x = np.mean(points[:, 0])
                min_y = np.mean(points[:, 1])
                max_y = np.max(points[:, 1])

                dist = abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0])
                
                if self.close_lanes[0] > dist:
                    self.close_lanes[1] = self.close_lanes[0]
                    self.lanes[1] = self.lanes[0]
                    self.close_lanes[0] = dist
                    self.lanes[0] = [min_x, min_y]
                elif self.close_lanes[1] > dist:
                    self.close_lanes[1] = dist
                    self.lanes[1] = [min_x, min_y]

            midpoint_x = int((self.lanes[0][0] + self.lanes[1][0]) / 2 * (frame.shape[1] / mask_width))
            midpoint_y = int((self.lanes[0][1] + self.lanes[1][1]) / 2 * (height / ll_seg_mask.shape[0]))
            
            # Koordinat düzeltmeleri
            x_new = int(midpoint_x * 1280 / 640) 
            y_new = int((midpoint_y * 720 / 640) * 2)

            # Orta noktaya beyaz daire çiz
            cv2.circle(frame, (x_new, y_new), 10, (255, 255, 255), -1)

        except Exception as e:
            # Şerit bulunamazsa veya hata olursa akışı bozma, devam et
            # print(f"Lane Error: {e}") 
            pass
            
        return frame

# ---------------------------------------------------------
# ANA FONKSIYON
# ---------------------------------------------------------
def main():
    # AYARLAR
    VIDEO_PATH = "input_video.mp4"       # Giriş videosu
    MODELS_DIR = "models"                # Modellerin olduğu klasör
    OUTPUT_PATH = f"output_{int(time.time())}.avi"
    
    # 1. Modelleri Başlat
    try:
        lane_detector = LaneDetector(os.path.join(MODELS_DIR, 'yolopv2.pt'))
        sign_detector = SignObjectDetector(MODELS_DIR)
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        print("Lütfen 'models' klasöründe .pt ve .pth dosyalarının tam olduğundan emin olun.")
        return

    # 2. Videoyu Başlat
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Hata: {VIDEO_PATH} açılamadı.")
        return

    # Çıktı Boyutları
    out_width = 1280
    out_height = 720
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (out_width, out_height))

    print("[INFO] İşlem başlıyor... Çıkmak için 'q' basın.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

        # Frame'i standart boyuta getir
        frame = cv2.resize(frame, (out_width, out_height))

        # --- AŞAMA 1: Şerit Takibi ---
        # Önce şeritleri çiziyoruz ki yazılar şeridin altında kalmasın
        frame = lane_detector.process(frame)

        # --- AŞAMA 2: Levha ve Nesne Tespiti ---
        frame = sign_detector.process(frame)

        # --- Gösterim ve Kayıt ---
        cv2.imshow("OTAGG Otonom Surus Entegrasyon", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video kaydedildi: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()