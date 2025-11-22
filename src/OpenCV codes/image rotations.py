import cv2
import numpy as np

path=r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg"
img=cv2.imread(path,0)

row,col=img.shape
#print(row)
#print(col)

m=np.float32([[1,0,100],[0,1,100]])
dst=cv2.warpAffine(img,m,(row,col))
cv2.imshow("dst",dst)
#resimlerde kaydırma işlemi
#kaydırılacağı koordinatları tanımlamak için
#[[1,0,x],[0,1,y]] matrisi kullanılır.
#warpAffine fonksiyonu ile de döndürme işlemi gerçekleştirilir

n=cv2.getRotationMatrix2D((row/2,col/2),180,1)
dst2=cv2.warpAffine(img,n,(row,col))
cv2.imshow("dst2",dst2)
#resimlerde döndürme işlemi 
#getRotationMatrix2D fonksiyonu 3 argüman alır
#1. ne kadar satır ve sütuna yerleştirileceği
#döndürülecek derece
#resmin boyutu: 1-olduğu gibi,2-2x yaklaştırır,3-3x yaklaştırır vb.

cv2.waitKey(0)
cv2.destroyAllWindows()

#bu işlemleri yapabilmek için resimler siyah beyaz olmalı
