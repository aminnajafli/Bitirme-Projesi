"""
Video Dosyası Üzerinde Nesne ve Levha Tespiti
=============================================

Bu kod, bir video dosyasını okur, üzerinde YOLO ile nesne ve trafik levhası
tespiti yapar ve sonuçları ekranda gösterir.

TÜM ARAÇ KONTROLÜ VE ARDUINO İLETİŞİM KODLARI KALDIRILMIŞTIR.
"""

import time
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
import numpy as np

# --- ARAÇ KONTROLÜ İLE İLGİLİ KISIMLAR TAMAMEN KALDIRILDI ---
# import socket
# class ArduinoSocketClient...
# def send_command_to_arduino_socket...
# def send_bldc_speed_command...
# def send_stepper_angle_command...
# def send_actuator_command...
# def send_system_emergency_stop...
# def emergency_stop_all...
# def process_traffic_signs_and_control...
# def control_vehicle_movement...
# def get_selected_port...
# -----------------------------------------------------------


# Trafik levhası sınıf isimleri (43 sınıf)
class_names = [
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

# MicronNet sınıflandırıcı
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

# --- ARAÇ KONTROLÜ İLE İLGİLİ FONKSİYONLAR KALDIRILDI ---


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılan cihaz: {device}")

    # Modelleri yükle
    yolo_traffic = YOLO("yolov12best.pt")
    yolo_obj = YOLO("object.pt")
    class_model = Net().to(device)
    class_model.load_state_dict(torch.load("balanced_data_micronnet.pth", map_location=device))
    class_model.eval()

    # --- Video dosyasını aç ---
    # video_path = "input_video.mp4" # !!! LÜTFEN BU YOLU DEĞİŞTİRİN !!!
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print(f"Hata: Video dosyası açılamadı: {video_path}")
    #     return

    camera_index = 1
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    if not cap.isOpened():
        print(f"Hata: Kamera açılamadı! İndeksi (0 veya 1) değiştirip tekrar deneyin.")

    # Levha sınıflandırması için görüntü dönüşümü
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # --- ŞERİT TESPİTİ MODELİ VE KODLARI KALDIRILDI ---

    cv2.namedWindow("Sonuc", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sonuc", 1280, 720)
    
    print("Video işleniyor... Çıkmak için 'q' tuşuna basın.")

    # --- Ana döngü video okuma üzerine kurulu ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya okuma hatası.")
            break # Video bittiğinde döngüden çık
        
        try:
            # Frame'i hazırla
            if frame is None:
                continue
            frame = np.array(frame)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            frame_for_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Frame dönüştürme hatası: {e}")
            continue


        # --- 1. Trafik levhası tespiti ve sınıflandırma ---
        try:
            results = yolo_traffic(frame_for_pil)
            # sign_list = [] # Sadece tespit için tutuluyordu, artık gerekmeyebilir
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cropped = frame_for_pil[b[1]:b[3], b[0]:b[2]]

                    if cropped.size == 0:
                        continue

                    cropped = np.asarray(cropped, dtype=np.uint8, order='C').view(np.ndarray)
                    img_pil = PILImage.fromarray(cropped)
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = class_model(img_tensor)
                        _, pred = torch.max(output, 1)
                        label = class_names[pred.item()]

                    # sign_list.append({"sign_name": pred.item()})

                    # Tespitleri frame üzerine çiz
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2) # Kırmızı kutu
                    cv2.putText(frame, label, (b[0], b[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            print(f"Trafik levhası hatası: {e}")

        # --- 2. İnsan/Araç Tespiti (YOLO Obj) ---
        try:
            results_obj = yolo_obj(frame_for_pil)
            labels = ["insan","bisiklet", "motor", "araba", "otobus", "kamyon"]
            
            for r in results_obj:
                for box in r.boxes:
                    cls_id = int(box.cls.cpu().numpy())
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    label = labels[cls_id]

                    # Tespitleri frame üzerine çiz
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2) # Mavi kutu
                    cv2.putText(frame, label, (b[0], b[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Sadece insan tespit edilirse ek bir uyarı ver
                    if label == "insan": 
                        cv2.putText(frame, "Dikkat! Insan tespiti!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        print(f"İnsan tespit edildi!")
                    

            # --- GÜVENLİK KİLİDİ YÖNETİMİ KALDIRILDI ---
            
        except Exception as e:
            print(f"İnsan/Araç tespiti hatası: {e}")

        
        # --- ŞERİT TAKİBİ VE ARAÇ KONTROL BLOKLARI KALDIRILDI ---
        
        # --- GÖRÜNTÜYÜ GÖSTER ---
        cv2.imshow("Sonuc", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Döngü bittiğinde kaynakları serbest bırak ---
    cap.release()
    cv2.destroyAllWindows()
    # --- ARDUINO BAĞLANTISI KAPATMA KISMI KALDIRILDI ---
    print("İşlem tamamlandı.")


if __name__ == '__main__':
    main()
