# Otonom Araçlarda Şerit Takibi ve Levha Tespiti Uygulamaları
2025-2026 Akademik Eğitim Yılı için Tasarım ve Bitirme Projesi dersleri için oluşturulmuş depo
                                            ---
 A repository created for the Design and Final Year Project courses for the 2025-2026 Academic Year                                         
                                            ***
Projede emeği geçenler: 032290094-Amin Najafli, 032290015-Yusuf Diler
Bursa Uludağ Üniversitesi Mühendislik Fakültesi Bilgisayar Mühendisliği Bölümü 

Bu proje, otonom araçlar için şerit takip algoritmalarının ve trafik levhalarının tespitini yapan yapay zeka modelinin tasarımını göstermektedir. Projede ağırlıklı olarak Python yazılım dili kullanılmaktadır. 

Projede yapılacaklar:
-gerçek zamanlı nesne ve levha tespiti
-simülasyon ortamında nesne ve levha tespiti
-şerit takibi için OpenCV tabanlı model
-YOLO modelleriyle eğitim ve karşılaştırmalı mertik değerlendirmeleri
                                            ---
Deponun İçeriği:
-dataset

-"src/" dosyası
  -OpenCV kaynak kodları
  -YOLO modeli kodları
  -testing.py dosyası
  -şerit takibi modülü
  -levha tespiti modülü
  -modelin fiziksel çalışmasını gösteren kodaz
-"results/" dosyası
  -metrik grafikleri
  -karmaşıklık matrisi çıktıları
  -test video ve ekran görüntüleri
  -şerit takibi ve nesne tespiti modelleri çıktıları
-gereksinimler dosyası

                                            ---
KURULUMLAR

Gazebo kurulumu:
```bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg
```
```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```

İndirmeler bittiğinde Gazebo kurulmuş olacak.

Kaynaktan kurulum yaparken Gazebo sürümünün belirtilmesi gerekmektedir:
```bash
export GZ_VERSION=harmonic
```

Kaynaktan kurulum yapılmasından dolayı paketin çalışma alanında izole olması önerilmektedir. Bundan dolayı home dizini içerisine ros2_ws'den farklı bir dizin oluşturmak gerekmektedir. Bu yapıldıktan sonra dizin içerisindeki src'ye ros_gz deposunun humble branch'ı klonlanmalıdır:
```bash
cd ~
mkdir -p ~/pkgs_ws/src
cd ~/pkgs_ws/src
git clone https://github.com/gazebosim/ros_gz.git -b humble
```

Kurulumu yapabilmek için ~/pkgs_ws dizini içerisine gelip rosdep ile bağımlılıkları kurmak ve colcon build ile kurulumu başlatmak gerekmektedir:

```bash
cd ~/pkgs_ws
rosdep install -r --from-paths src -i -y --rosdistro humble
colcon build
```

Bütün bu işlemler bittikten sonra pkgs_ws'nin bashrc'den source yapılması gerekmektedir:
```bash
echo "source ~/pkgs_ws/install/setup.bash" >> ~/.bashrc
```

İşlemler sonucunda ros_gz paketi kurulumu tamamlanmış demektir.
