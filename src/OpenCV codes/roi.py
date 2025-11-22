#roi ->region of interest
import cv2
import numpy as np
import matplotlib.pyplot as plt

path=r"C:\Users\Amin\opencv_udemy\coreOperations\testImages\galatasaray-2845588_640.jpg"
img=cv2.imread(path)

roi=img[200:300,200:300]
img[300:400,300:400]=roi #kesilen kısmı fotoğrafa yapıştırıyoruz
#aralıkları aynı olmalı atanılacak kısımla kesilen kısmın
cv2.imshow("face", roi)
#print(img.shape)
cv2.imshow("basketball", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


