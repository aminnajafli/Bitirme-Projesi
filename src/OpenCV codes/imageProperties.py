import cv2
import numpy as np
import matplotlib.pyplot as  plt

img=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\testImages\opencv.png")

print(img.shape) #görselin genişliğini, yüksekliğini ve kanal sayısını verir
#yükseklik-y ekseni
#genişlik-x ekseni
"""kanal sayısı yerindeki sayı görselin siyah beyaz, gri ya  da renkli olması hakkında bilgi verir.
eğer çıktıdaki sayı 3 ise, görsel renkli
                    1 ya da boş ise, gri"""

#img.size->görselin boyutunu verir.

cv2.imshow("openCV", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
