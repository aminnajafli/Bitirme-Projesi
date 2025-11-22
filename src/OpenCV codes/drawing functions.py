import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas=np.zeros((512,512,3),dtype=np.uint8)+255
cv2.line(canvas, (50,50),(512,512),(255,0,0),thickness=5) 
#doğru çizmek için kullanılan bir fonksiyondur
#parametreleri:
#1. çizim yapılacak tuvalin adı
#2. doğrunun başlangıç noktası
#3. doğrunun bitiş noktasaı
#4. çizilecek doğrunun rengi
#5. doğrunun kalınlığı
cv2.line(canvas, (100,50),(200,250),(0,0,255),7)
cv2.rectangle(canvas,(444,117),(117,444),(0,255,0),2)
#dikdörtgen çizmek için kullanılır parametrelerin mantığı aynı
cv2.rectangle(canvas,(400,100),(500,1),(117,2,117),-1)
#kalınlık değeri -1 girildiğinde dikdörtgenin içi dolu olur
cv2.circle(canvas,(250,250),100,(44,77,74),4)
#çember çizmek için kullanılır
#parametreleri:
#1. çizim yapılacak tuvalin adı
#2. çemberin merkez noktası
#3. çemberin yarıçap uzunluğu
#4. çizilecek çemberin rengi
#5. çemberin kalınlığı
"""                 ---                 """
#üçgen çizimi
#opencv üçgenler için ayrıca fonksiyon tanımlamıyor
p1=(100,200)
p2=(50,50)
p3=(300,300)
cv2.line(canvas, p1,p2,(0,0,0),10)
cv2.line(canvas, p2,p3,(0,0,0),10)
cv2.line(canvas, p1,p3,(0,0,0),10)
#bu yüzden 3 tane doğru koordinatları birleştirilir.


print(canvas)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
