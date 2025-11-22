import cv2
import numpy as np
import matplotlib.pyplot as plt

path=r"C:\Users\Amin\opencv_udemy\coreOperations\testImages\opencv.png"
img=cv2.imread(path)

print(img.shape)
(b,g,r)=cv2.split(img)


cv2.imshow("openCV",img)
cv2.imshow("openCV-b",b) #görselden mavi kısmın çıkarılmış hali
cv2.imshow("openCV-g",g) #görselden yeşil kısmın çıkarılmış hali
cv2.imshow("openCV-r",r) #görselden kırmızı kısmın çıkarılmış hali
cv2.waitKey(0)
cv2.destroyAllWindows()
