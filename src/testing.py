"""
Arduino Motor Control System Integration
========================================

Bu dosya, PlatformIO tabanlı Arduino motor kontrol sistemi ile entegre edilmiştir.

Arduino Komut Yapısı:
- M,FORWARD,speed veya M,REVERSE,speed: BLDC motor kontrolü (linearX için)
- S,ANGLE,degrees: Stepper motor açı kontrolü (linearZ için)
- S,position: Stepper motor pozisyon kontrolü
- S,STOP: Stepper motor durdurma
- A,command: Actuator kontrolü (b=fren, o=açma, c=kapatma, vb.)
- SYS,EMERGENCY: Sistem acil durdurma
- SYS,RESET: Sistem reset
- SYS,ENABLE: Sistem etkinleştirme

Kontrol Mantığı:
- linearX: Araç hızı (pozitif=ileri, negatif=geri, 0=dur)
- linearZ: Direksiyon kontrolü (pozitif=sağa, negatif=sola, 0=merkez)
"""

import time
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from simple_pid import PID
import numpy as np
from image_processing_2025.utils import letterbox, select_device, lane_line_mask
import pyzed.sl as sl
from filter import Filter
from sliding_windows.SlidingWindow import LaneDetector
from sliding_windows.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, letterbox
# import /home/eren/Desktop/gomulu_sistem_2025/arayuz_old.py
import sys
import socket


# --- Arduino ile sürekli bağlantı için Socket Client Sınıfı ---
class ArduinoSocketClient:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Arduino'ya bağlantı kuruldu: {self.host}:{self.port}")
        except Exception as e:
            print(f"Arduino bağlantısı kurulamadı: {e}")
            self.sock = None

    def send_command(self, command):
        try:
            if self.sock is None:
                self.connect()
            if self.sock:
                # Noktalı virgül ile birden fazla komut gönderimi desteği
                if isinstance(command, list):
                    command = ';'.join(str(c).rstrip(';\r\n') for c in command)
                else:
                    command = str(command).rstrip(';\r\n')
                command += '\n'
                self.sock.sendall(command.encode('utf-8'))
                print(f"Arduino'ya komut gönderildi: {command.strip()}")
                time.sleep(0.01)  # 10ms bekle
        except Exception as e:
            print(f"Komut gönderilemedi, bağlantı sıfırlanıyor: {e}")
            self.close()
            self.connect()
            if self.sock:
                try:
                    self.sock.sendall(command.encode('utf-8'))
                    print(f"Arduino'ya komut tekrar gönderildi: {command.strip()}")
                    time.sleep(0.01)
                except Exception as e2:
                    print(f"Komut tekrar da gönderilemedi: {e2}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
                print("Arduino bağlantısı kapatıldı.")
            except Exception as e:
                print(f"Bağlantı kapatılamadı: {e}")
            self.sock = None


# Global olarak bir ArduinoSocketClient örneği oluştur
arduino_client = ArduinoSocketClient()

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


# --- Port paylaşımı için seçili portu oku ---
def get_selected_port(filename="/home/eren/Desktop/gomulu_sistem_2025/selected_port.txt"):
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Port dosyası okunamadı: {e}")
        return None


# --- Komutları Arduino'ya göndermek için fonksiyon ---
def send_command_to_arduino_socket(command, host='127.0.0.1', port=8888):
    # host ve port parametreleri artık kullanılmıyor, backward compatibility için tutuldu
    arduino_client.send_command(command)


# --- PlatformIO Arduino komut fonksiyonları ---
def send_bldc_speed_command(speed, direction="FORWARD"):
    """BLDC motor hız kontrolü - linearX için"""
    if direction == "FORWARD":
        command = f"M,FORWARD,{abs(speed)}"
    else:
        command = f"M,REVERSE,{abs(speed)}"
    send_command_to_arduino_socket(command)


def send_stepper_angle_command(angle):
    """Stepper motor açı kontrolü - linearZ için"""
    angle = max(min(angle, 2160), -2160)  # Açı sınırlandırma
    command = f"S,{angle}"
    if angle != 0:
        send_command_to_arduino_socket(command)


def send_stepper_position_command(position):
    """Stepper motor pozisyon kontrolü"""
    command = f"S,{position}"
    send_command_to_arduino_socket(command)


def send_stepper_stop_command():
    """Stepper motor durdurma"""
    command = "S,STOP"
    send_command_to_arduino_socket(command)


def send_actuator_command(actuator_cmd):
    """Actuator kontrolü"""
    command = f"A,{actuator_cmd}"
    send_command_to_arduino_socket(command)


def send_system_emergency_stop():
    """Sistem emergency stop"""
    command = "M,0"
    send_command_to_arduino_socket(command)


def send_system_reset():
    """Sistem reset"""
    command = "M,70"
    send_command_to_arduino_socket(command)


def send_system_enable():
    """Sistem etkinleştir"""
    command = "SYS,ENABLE"
    send_command_to_arduino_socket(command)


# --- Stepper motor referans noktası fonksiyonları ---
def calibrate_stepper_reference(linearZ_value, calibration_samples, max_samples=10):
    """Stepper motor için referans noktası kalibrasyonu"""
    if len(calibration_samples) < max_samples:
        calibration_samples.append(linearZ_value)
        print(f"Kalibrasyon örneği eklendi: {linearZ_value:.2f} ({len(calibration_samples)}/{max_samples})")
        return False, 0  # Henüz kalibre edilmedi
    else:
        # Ortalama hesapla ve referans noktası olarak kullan
        reference_center = sum(calibration_samples) / len(calibration_samples)
        print(f"Stepper motor referans noktası kalibre edildi: {reference_center:.2f}")
        return True, reference_center


def calculate_stepper_correction(current_linearZ, reference_center, stepper_position_offset):
    """Stepper motor için düzeltme hesaplama"""
    deviation_from_center = current_linearZ - reference_center
    corrected_position = deviation_from_center + stepper_position_offset
    return deviation_from_center, corrected_position


def reset_stepper_to_reference():
    """Stepper motor'u referans noktasına sıfırla"""
    send_stepper_position_command(0)  # Pozisyon sıfırla
    print("Stepper motor pozisyonu referans noktasına sıfırlandı")


# --- Mutlak açı komutu gönderme fonksiyonu ---
def send_absolute_angle_command(angle):
    """Step motor'u mutlak açıya döndürme komutu gönder"""
    try:

        send_stepper_angle_command(angle)
        print(f"Mutlak açı komutu gönderildi: {angle:.2f} derece")
    except Exception as e:
        print(f"Mutlak açı komutu gönderilemedi: {e}")


# --- LinearX ve LinearZ kontrolü için yeni fonksiyonlar ---
def control_vehicle_movement(linearX, linearZ):
    """Aracın hareket kontrolü - linearX: hız, linearZ: yönlendirme"""
    # LinearX -> BLDC motor hız kontrolü
    if linearX > 0:
        speed = int(abs(linearX) * 50)
        speed = min(speed, 70)
        send_bldc_speed_command(speed, "FORWARD")
    elif linearX < 0:
        speed = int(abs(linearX) * 50)
        speed = min(speed, 70)
        send_bldc_speed_command(speed, "REVERSE")
    else:
        send_bldc_speed_command(0, "FORWARD")

    # LinearZ -> Stepper motor pozisyon kontrolü (angle yerine position gönder)
    if abs(linearZ) > 0.5:
        # Pozisyona çevir (ör: linearZ * 10 ile çarp, ihtiyaca göre değiştir)
        position = int(linearZ * 10)
        print(f"Stepper pozisyon komutu gönderiliyor: {position}")
        send_stepper_angle_command(linearZ)
        # else:
        send_stepper_angle_command(0)


def process_traffic_signs_and_control(sign_list, linearX, linearZ):
    """Trafik levhalarını analiz et ve araç kontrolü için linearX ve linearZ değerlerini ayarla"""

    for detected in sign_list:
        sign_name = detected['sign_name']
        distance = detected['distance']

        print(f"Traffic Sign: {class_names[sign_name]} - Distance: {distance:.1f} cm")

        # Dur işareti (Class 14)
        if sign_name == 14 and distance < 300:  # 3 metre yakınsa
            print("DUR işareti tespit edildi - Araç durduruluyor")
            linearX = 0.0
            linearZ = 0.0
            send_actuator_command("b")  # Fren uygula
            return linearX, linearZ, True  # stop_detected=True

        # Kırmızı ışık (Class 32)
        elif sign_name == 32 and distance < 500:  # 5 metre yakınsa
            print("Kırmızı ışık tespit edildi - Araç durduruluyor")
            linearX = 0.0
            linearZ = 0.0
            send_actuator_command("b")  # Fren uygula
            return linearX, linearZ, True

        # Yeşil ışık (Class 41) - İleri git
        elif sign_name == 41 and distance < 500:
            print("Yeşil ışık tespit edildi - İleri hareket")
            linearX = 1.5
            linearZ = 0.0

        # Sağa dönüş levhaları (Class 34, 36, 39)
        elif sign_name in [34, 36, 39] and distance < 400:
            print("Sağa dönüş işareti tespit edildi")
            linearX = 0.8
            linearZ = 15.0  # Sağa dönüş için pozitif değer

        # Sola dönüş levhaları (Class 33, 37, 38)
        elif sign_name in [33, 37, 38] and distance < 400:
            print("Sola dönüş işareti tespit edildi")
            linearX = 0.8
            linearZ = -15.0  # Sola dönüş için negatif değer

        # Hız sınırları (Class 0-7)
        elif sign_name in [0, 1, 2, 3, 4, 5, 6, 7] and distance < 400:
            speed_limits = [20, 30, 50, 60, 70, 80, 80, 100]  # km/h
            max_speed = speed_limits[sign_name]
            # Hız sınırına göre linearX'i ayarla (1.0 = ~50 km/h kabul edelim)
            linearX = min(linearX, max_speed / 50.0)
            print(f"Hız sınırı {max_speed} km/h tespit edildi - Hız ayarlandı: {linearX:.1f}")

        # Yol çalışması (Class 23) - Yavaşla
        elif sign_name == 23 and distance < 400:
            print("Yol çalışması tespit edildi - Yavaşlama")
            linearX = min(linearX, 0.5)  # Maksimum yarı hız

        # Kaygan yol (Class 24) - Yavaşla
        elif sign_name == 24 and distance < 400:
            print("Kaygan yol tespit edildi - Yavaşlama")
            linearX = min(linearX, 0.7)  # Yavaşla

        # Dönel kavşak (Class 25) - Yavaşla
        elif sign_name == 25 and distance < 400:
            print("Dönel kavşak tespit edildi - Yavaşlama")
            linearX = min(linearX, 0.6)

    return linearX, linearZ, False  # stop_detected=False


# --- Modelinizden gelen angular veriyi işleme ---
def process_model_angular_data(angular_value):
    """Modelden gelen angular veriyi işle ve Arduino'ya gönder"""
    global current_target_angle, last_angle_command_time

    try:
        # Angular veriyi float'a çevir
        angle = float(angular_value)

        current_target_angle = angle
        last_angle_command_time = time.time()
        send_absolute_angle_command(angle)
        print(f"Model angular verisi işlendi: {angle:.2f} derece")

    except (ValueError, TypeError) as e:
        print(f"Angular veri işleme hatası: {e}")


# --- Direkt model entegrasyonu ---
def handle_model_angular_output(model_output):
    """Model çıktısından angular veriyi al ve işle"""
    # Modelinizin çıktı formatına göre adapte edin
    # Örnek: model_output içinde 'angle' anahtarı varsa
    if hasattr(model_output, 'angle') or 'angle' in model_output:
        angle = model_output.get('angle', 0.0) if isinstance(model_output, dict) else model_output.angle
        process_model_angular_data(angle)
    else:
        # Eğer model_output doğrudan açı değeriyse
        process_model_angular_data(model_output)


# --- Global değişkenler ---
current_target_angle = None
last_angle_command_time = 0


def emergency_stop_all():
    """Tüm motorları acil durumda durdur"""
    send_system_emergency_stop()
    send_actuator_command("b")  # Fren uygula
    print("Acil durum - Tüm motorlar durduruldu!")


def main():
    global current_target_angle, last_angle_command_time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Modelleri yükle
    yolo_traffic = YOLO("alltoget_v2.pt")  # .to(device) yapma, ultralytics kendi halleder
    yolo_obj = YOLO("object.pt")  # .to(device) yapma, ultralytics kendi halleder
    class_model = Net().to(device)
    class_model.load_state_dict(torch.load("balanced_data_micronnet.pth", map_location=device))
    class_model.eval()

    pid = PID(0.005, 0, 0.001)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("ZED açılmadı:", status)
        exit(1)

    image = sl.Mat()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    weights = 'yolopv2.pt'
    device_lane = select_device('0')
    lane_model = torch.jit.load(weights).to(device_lane)
    lane_model.eval()
    half = True
    if half:
        lane_model.half()

    cv2.namedWindow("Sonuç", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sonuç", 1280, 720)

    # --- Video Recorder ---
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"videos/videos_{time.time()}.avi", fourcc, 20.0, (1280, 720))

    # --- Şerit takibi komut zamanlayıcısı ---
    last_command_time = time.time()
    command_interval = 0.15  # 150ms aralıklarla komut gönder (daha yavaş hareket için)

    # --- Stepper motor referans noktası sistemi ---
    stepper_reference_center = 0  # Merkez referans noktası
    stepper_position_offset = 0  # Mevcut pozisyon offset'i
    reference_calibrated = False  # Referans kalibre edildi mi?
    calibration_samples = []  # Kalibrasyon örnekleri
    max_calibration_samples = 10  # Maksimum kalibrasyon örnek sayısı

    # --- İnsan tespit durumu ve güvenlik kilidi ---
    human_detected = False  # İnsan tespit edildi mi?
    safety_locked = False  # Güvenlik kilidi (manuel olarak sıfırlanmalı)
    last_human_detection_time = 0  # Son insan tespit zamanı
    safety_timeout = 5.0  # 5 saniye boyunca insan görünmezse kilit açılır

    # --- Arduino portunu aç ---
    selected_port = get_selected_port()
    if selected_port:
        print(f"Seçili port: {selected_port}")
        print("Arduino bağlantısı socket üzerinden yapılacak")
    else:
        print("Uygun port bulunamadı! Arduino arayüzünü başlatın.")

    while True:  # Her frame'de sıfırla  
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Kesin np.ndarray ve doğru tip yap
            frame = np.array(frame)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame_for_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Trafik levhası tespiti ve sınıflandırma ---
            try:
                results = yolo_traffic(frame_for_pil)
                sign_list = []
                for r in results:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        cropped = frame_for_pil[b[1]:b[3], b[0]:b[2]]

                        if cropped.size == 0:
                            continue

                        # Force base ndarray for PIL
                        cropped = np.asarray(cropped, dtype=np.uint8, order='C').view(np.ndarray)
                        img_pil = PILImage.fromarray(cropped)
                        img_tensor = transform(img_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = class_model(img_tensor)
                            _, pred = torch.max(output, 1)
                            label = class_names[pred.item()]

                        # Calculate center of bounding box
                        center_x = int((b[0] + b[2]) / 2)
                        center_y = int((b[1] + b[3]) / 2)

                        # Get depth map from ZED
                        depth_map = sl.Mat()
                        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                        depth_value = depth_map.get_value(center_x, center_y)[1]  # (err, value)
                        if np.isnan(depth_value) or depth_value <= 0:
                            depth_cm = 0.0
                        else:
                            depth_cm = float(depth_value) / 10  # meters to centimeters

                        # Çizim için frame OpenCV BGR formatında ve contiguous olmalı:
                        if not isinstance(frame, np.ndarray):
                            frame = np.array(frame)
                        if not frame.flags['C_CONTIGUOUS']:
                            frame = np.ascontiguousarray(frame)
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        # sign_list.append({"sign_name": pred.item(), "distance": depth_cm})
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cv2.putText(frame, label, (b[0], b[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"Depth: {depth_cm:.1f} cm", (center_x - 100, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            except Exception as e:
                print(f"Trafik levhası hatası: {e}")

            try:
                results_obj = yolo_obj(frame_for_pil)
                labels = ["insan", "bisiklet", "motor", "araba", "otobus", "kamyon"]
                human_detected_this_frame = False  # Bu frame'de insan tespit edildi mi?

                for r in results_obj:
                    for box in r.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        label = labels[cls_id]

                        # Calculate center of bounding box
                        center_x = int((b[0] + b[2]) / 2)
                        center_y = int((b[1] + b[3]) / 2)

                        # Get depth map from ZED
                        depth_map = sl.Mat()
                        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                        depth_value = depth_map.get_value(center_x, center_y)[1]  # (err, value)
                        if np.isnan(depth_value) or depth_value <= 0:
                            depth_cm = 0.0
                        else:
                            depth_cm = float(depth_value) / 10  # meters to centimeters

                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)  # Mavi kutu
                        cv2.putText(frame, label, (b[0], b[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, f"Depth: {depth_cm:.1f} cm", (center_x - 100, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        if label == "insan" and depth_cm < 300:
                            human_detected_this_frame = True  # Bu frame'de insan tespit edildi
                            human_detected = True  # Genel insan tespit durumu
                            safety_locked = True  # Güvenlik kilidi aktif
                            last_human_detection_time = time.time()  # Son tespit zamanını kaydet
                            cv2.putText(frame, "Dikkat! Yakın insan tespiti!", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            print(f"İnsan tespit edildi! Mesafe: {depth_cm:.1f} cm")
                            # İnsan tespit edildiğinde tüm motorları durdur
                            emergency_stop_all()
                            print("Tüm motorlar durduruldu - İnsan tespit edildi!")

                            send_system_reset()

                # Güvenlik kilidi yönetimi
                current_time = time.time()
                if safety_locked and not human_detected_this_frame:
                    # İnsan bu frame'de görünmedi ama güvenlik kilidi aktif
                    if current_time - last_human_detection_time > safety_timeout:
                        # Belirli süre geçti, güvenlik kilidini kaldır
                        safety_locked = False
                        human_detected = False
                        send_system_reset()
                        print(f"Güvenlik kilidi kaldırıldı - {safety_timeout} saniye boyunca insan görünmedi")
                    else:
                        # Henüz güvenlik süresi geçmedi, sistem kilitli kalıyor
                        remaining_time = safety_timeout - (current_time - last_human_detection_time)
                        human_detected = True  # Sistem kilitli kalıyor
                        print(f"Güvenlik kilidi aktif - Sistem {remaining_time:.1f} saniye daha kilitli kalacak")
                elif not safety_locked:
                    # Güvenlik kilidi yok, normal çalışma
                    human_detected = human_detected_this_frame
            except Exception as e:
                print(f"İnsan/Araç tespiti hatası: {e}")

            # --- Şerit takibi ---
            try:
                img = letterbox(frame, 640, stride=32)[0]

                if img.shape[2] == 4:
                    img = img[:, :, :3]

                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device_lane)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                _, _, ll = lane_model(img)
                ll_seg_mask = lane_line_mask(ll)
                ll = ll.squeeze().detach().cpu().numpy()

                ll = ll.astype(np.uint8)

                if ll.shape != frame.shape[:2]:
                    ll = cv2.resize(ll, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                height, width = img.shape[2:]
                foreground = ll == 1  # veya ll == 255, çıktına göre
                # lane_lines = np.zeros((ll_seg_mask.shape[0], ll_seg_mask.shape[1], 1), dtype=np.uint8)
                # lane_lines[ll_seg_mask == 1] = 255
                # temp = cv2.resize(lane_lines, (height, width), cv2.INTER_CUBIC)
                # filter = Filter(width, height)  # Filter sınıfını başlatıyoruz
                # detector = LaneDetector(width, height)  # LaneDetector sınıfını başlatıyoruz
                ## Perspektif dönüşümünü gerçekleştir
                # transformed_frame = filter.perspective_transform(temp)
                ## Şerit takibi yapılması
                # detected_lanes = detector.detect_lanes(transformed_frame)
                # if detected_lanes is None:
                #    continue
            #
            # left_base, right_base = detected_lanes
            # lane_mask, lx, rx, vehicle_position = detector.sliding_window_tracking(transformed_frame, left_base, right_base)
            #
            # original_lane_positions = filter.reverse_perspective_transform(lane_mask) #ters perspektif dönüşümü
            #
            # if vehicle_position is not None:
            #    print(f"Aracın Konumu: {vehicle_position:.2f}")

            # # Lane maskesinin gösterilmesi
            # cv2.imshow("Lane Detection - Sliding Windows", lane_mask)
            # cv2.imshow("Transformed Frame", transformed_frame)
            # mask_copy, lx, rx, vehicle_position = detector.sliding_window_tracking(transformed_frame, left_base, right_base)
            # cv2.imshow("Sliding Window", mask_copy)  # Sliding window takibini göster
            # cv2.imshow("Reversed Perspective Frame", original_lane_positions)

            except Exception as e:
                print(f"Şerit takibi hatası: {e}")

            linearX = 0.0
            linearZ = 0.0
            stop_sign_detected = False

            # Trafik levhalarını analiz et ve kontrol değerlerini ayarla
            # if sign_list:
            # linearX, linearZ, stop_sign_detected = process_traffic_signs_and_control(sign_list, linearX, linearZ)

            # Dur işareti tespit edilmişse diğer kontrolleri atla
            if stop_sign_detected:
                print("Dur işareti nedeniyle araç durduruldu")
                control_vehicle_movement(0.0, 0.0)  # Araç durdur
                continue  # Bu frame'i atla, bir sonraki frame'e geç
            try:
                contours, _ = cv2.findContours((ll_seg_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                lanes, close_lanes = [[0, 0], [0, 0]], [999, 999]
                mask_width = ll_seg_mask.shape[1]
                mean = mask_width / 2

                for contour in contours:
                    if cv2.contourArea(contour) < 200:
                        continue
                    points = contour[:, 0, :]
                    if len(contour) < 20:
                        continue
                    print(len(contour))
                    min_x = np.mean(points[:, 0])
                    min_y = np.mean(points[:, 1])
                    max_y = np.max(points[:, 1])
                    if close_lanes[0] > abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0]):
                        close_lanes[1] = close_lanes[0]
                        lanes[1] = lanes[0]
                        close_lanes[0] = abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0])
                        lanes[0] = [min_x, min_y]
                    elif close_lanes[1] > abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0]):
                        close_lanes[1] = abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0])
                        lanes[1] = [min_x, min_y]

                midpoint_x = int((lanes[0][0] + lanes[1][0]) / 2 * (width / mask_width))
                midpoint_y = int((lanes[0][1] + lanes[1][1]) / 2 * (height / ll_seg_mask.shape[0]))
                x_new = int(midpoint_x * 1280 / 640)
                y_new = int((midpoint_y * 720 / 640) * 2)
                cv2.circle(frame, (x_new, y_new), 7, (255, 255, 255), -1)
                print(f"Midpoint: ({midpoint_x}, {midpoint_y}), width: {width}")
                control = -1 * ((width / 2) - midpoint_x)

                # send_absolute_angle_command(36*control)

                # # Şerit takibi için temel hareket kontrolü (trafik levhası kontrolü olmadığında)
                # if not stop_sign_detected and not sign_list:  # Dur işareti yok ve trafik levhası yok
                #     linearX = 2.0  # Temel ileri hız

                #     # LinearZ değerini sınırla ve yumuşat
                #     raw_linearZ = -float(control)
                #     linearZ = raw_linearZ  # Başlangıçta yumuşatma uygulanmadan önceki değer
                #     # 1. Değeri sınırla (-50 ile +50 arası)
                #     max_turn_value = 100000.0
                #     if raw_linearZ > max_turn_value:
                #         raw_linearZ = max_turn_value
                #     elif raw_linearZ < -max_turn_value:
                #         raw_linearZ = -max_turn_value

                #     # 2. Yumuşatma faktörü uygula (0.3 = %30 yeni değer, %70 eski değer)
                #     smoothing_factor = 0.3
                #     if 'previous_linearZ' not in globals():
                #         global previous_linearZ
                #         previous_linearZ = 0.0

                #     #linearZ = (smoothing_factor * raw_linearZ) + ((1 - smoothing_factor) * previous_linearZ)
                #     previous_linearZ = linearZ

                #     print(f"Control: {control:.1f}, Raw LinearZ: {raw_linearZ:.1f}, Smoothed LinearZ: {linearZ:.1f}")
                # elif sign_list and not stop_sign_detected:  # Trafik levhası var ama dur değil
                #     # Trafik levhası kontrolü zaten yukarıda yapıldı, sadece mevcut değerleri kullan
                #     print(f"Traffic sign control active - LinearX: {linearX:.1f}, LinearZ: {linearZ:.1f}")
                # else:
                #     # Dur durumunda sıfır değerler
                #     linearX = 0.0
                #     linearZ = 0.0

                print(f"Final control values - LinearX: {linearX:.1f}, LinearZ: {linearZ:.1f}")

                # === MODELİNİZDEN GELEN ANGULAR VERİSİ İÇİN ÖRNEK ENTEGRASYON ===
                # Burada modelinizden gelen angular veriyi alabilirsiniz
                # Örnek kullanım:
                # if model_angular_data_available:
                #     handle_model_angular_output(model_angular_data)

                # Eğer modeliniz sürekli angular verisi üretiyorsa, 
                # bu kısmı modelinizin çıktısına göre adapte edin
                # Örnek: angular_prediction = your_model.predict_angle(frame)
                # handle_model_angular_output(angular_prediction)

                # Stepper motor referans noktası kalibrasyonu
                # Şerit algılandığında Arduino'ya komut gönder (zamanlayıcı ile)
                current_time = time.time()
                if len(contours) > 0 and (current_time - last_command_time) > command_interval:
                    # İnsan tespit edildiğinde hiçbir motor komutu gönderme
                    if human_detected:
                        print("İnsan tespit edildi - Tüm motor komutları engellendi")
                        # Güvenlik için tekrar dur komutları gönder
                        emergency_stop_all()
                        last_command_time = current_time
                    else:
                        # İnsan tespit edilmediğinde normal çalışma
                        # Control vehicle movement using linearX and linearZ
                        control_vehicle_movement(70, 36 * control)
                        last_command_time = current_time

                # Şerit algılanmadığında tüm motorları durdur

                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame[foreground] = [0, 0, 255]

                # Referans noktası durumunu ekrana yazdır
                # if reference_calibrated:
                # cv2.putText(frame, f"Stepper Referans: {stepper_reference_center:.1f}",
                # (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #    if len(contours) > 0:
                #        deviation_from_center, _ = calculate_stepper_correction(
                #            linearZ, stepper_reference_center, stepper_position_offset)
                # cv2.putText(frame, f"Sapma: {deviation_from_center:.1f}",
                # (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                ##else:
                # cv2.putText(frame, f"Kalibrasyon: {len(calibration_samples)}/{max_calibration_samples}",
                # (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "Serit merkezinde kalip kalibre edin",
                #    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # İnsan tespit durumunu göster
                # if safety_locked:
                #    remaining_time = safety_timeout - (time.time() - last_human_detection_time)
                #    if remaining_time > 0:
                #        #cv2.putText(frame, f"GUVENLIK KILIDI - {remaining_time:.1f}s KALDI", 
                #           (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "Manuel sifirlama icin 'R' tusuna basin",
                #           (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                # else:
                #    cv2.putText(frame, "GUVENLIK KILIDI - SURESIZ KILITLI",
                #                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #    cv2.putText(frame, "Manuel sifirlama icin 'R' tusuna basin",
                #                (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                # elif human_detected:
                #    cv2.putText(frame, "INSAN TESPIT EDiLDi - W KOMUTU ENGELLENDI", 
                #                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # else:
                #    cv2.putText(frame, "INSAN TESPIT EDiLMEDi - NORMAL SEYIR", 
                #                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #
                ## Mutlak açı kontrol durumunu göster
                # if current_target_angle is not None:
                #    cv2.putText(frame, f"Hedef Aci: {current_target_angle:.1f} derece", 
                #                (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                # else:
                #    cv2.putText(frame, "Mutlak Aci Komutu Yok", 
                #                (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128sZ, 128, 128), 2)

            except Exception as e:
                print(f"Lane detection error: {e}")
                control = 0

            cv2.imshow("Sonuç", frame)

            out.write(cv2.resize(frame, (1280, 720)))

            # Klavye kontrolleri
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # 'r' tuşu ile güvenlik kilidini sıfırla
                safety_locked = False
                human_detected = False
                print("Güvenlik kilidi manuel olarak sıfırlandı - Sistem normal çalışmaya devam ediyor")
                # Sistemı etkinleştir ve hazır duruma getir
                send_system_enable()
                print("Sistem manuel olarak etkinleştirildi")

    zed.close()
    out.release()
    cv2.destroyAllWindows()
    # Program sonunda bağlantıyı kapat
    arduino_client.close()


if __name__ == "__main__":
    main()
