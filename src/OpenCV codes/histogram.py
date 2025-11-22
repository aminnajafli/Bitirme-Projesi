import cv2
import numpy as np
import matplotlib.pyplot as plt

#bir görüntünün histogramını çizmek

img=np.zeros((500,500),np.uint8)+200
cv2.rectangle(img,(0,60),(200,150),(113,250,200),-1)
cv2.rectangle(img,(250,170),(350,200),(179,0,254),-1)
cv2.imshow("img",img)

plt.hist(img.ravel(),256,[0,256])
plt.show()
#genelde elimizdeki görsel çok boyutludur
#bu yüzden ravel() fonksiyonu ile görselin oluşturduğu matris
#tek boyutlu (1D) bir diziye dönüştürülür
#yani bütün pikselleri tek bir liste haline getirir
#256-histogramın 256 sütundan oluşacağını belirtir
#0-256->piksel değerlerinin aralığını gösterir

cv2.waitKey(0)
cv2.destroyAllWindows()
