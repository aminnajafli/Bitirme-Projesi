import cv2
import numpy as np

path=r"C:\Users\Amin\opencv_udemy\ducky.jpeg"
img=cv2.imread(path)

img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow("duck",img)
cv2.imshow("duck2",img_rgb)
cv2.imshow("duck3",img_hsv)
cv2.imshow("duck4",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
