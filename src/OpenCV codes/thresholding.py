import cv2
import numpy as np 
import matplotlib.pyplot as plt

path=r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg"
img=cv2.imread(path,0)
#resim üzerinde thresholding yapabilmek için resim siyah beyaz olmalı

ret,th1=cv2.threshold(img,10,255,cv2.THRESH_BINARY)
#max threshold değeri 255'tir.
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)


cv2.imshow("img-th1",th1)
cv2.imshow("img-th2",th2)
cv2.imshow("img-th3",th3)
cv2.waitKey(0)
cv2.destroyAllWindows()
