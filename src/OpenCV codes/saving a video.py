#bilgisayar kamerasından canlı görüntü alıp ekranda gösteren opencv programı
import cv2

cap=cv2.VideoCapture(0) #kamera bağlantısı oluşturur. burda cap değişkeni kameradan
                        #gelen görüntüleri okuyabileceğimiz bir "video yakalayıcı" nesnedir.
"""VideoCapture() genel olarak kamera kaynağı veya bir video dosyasını açmak için kullanılır"""


fileName=r"C:\Users\Amin\opencv_udemy\videoKaydetme\webcam.avi"
codec=cv2.VideoWriter.fourcc('W','M','V','2') #videonun hangi sıkıştırma biçimiyle yazılacağını belirler (WMV2)
frameRate=30 #FPS- yazılacak videonun saniyedeki kare sayısı
resolution=(640,480)
"""kare boyutunu bildirir.
***yazıcıya verilen çözünürlük, yazılan frame'in gerçek boyutuyla
aynı olmalıdır, aksi halde write() başarısız olabilir"""

videoFileOutput=cv2.VideoWriter(fileName,codec,frameRate,resolution)
"""VideoWriter oluşturur(diske video kaydı için)"""

while True:
    ret,frame=cap.read() #kameradan bir frame döndürülür.
    """geriye iki değer döner, ret-> görüntü başarıyla okunduysa true, aksi halde false
    kameradan alınan tek karelik görüntü(numpy dizisi)"""

    #frame=cv2.flip(frame,0) """görüntüyü x eksenine göre tersini alarak gösterir."""
    #frame=cv2.flip(frame,1) """görüntüyü y eksenine göre ters çevirir."""
    #frame=cv2.flip(frame,-1) """görüntüyü orijine göre tersini alarak gösterir"""

    videoFileOutput.write(frame) #diske kare yazdırır. burda frame boyutu ve resolution boyutu aynı olmalı

    cv2.imshow("webcam live", frame)
    if cv2.waitKey(1)& 0xFF==ord("q"):
        break

videoFileOutput.release() #VideoWriter'ı kapatır ve dosyayı düzgün finalize eder
cap.release() #kamerayı serbest bırakır
cv2.destroyAllWindows()
