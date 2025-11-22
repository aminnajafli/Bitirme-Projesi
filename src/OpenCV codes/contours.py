import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
path=r"C:\Users\Amin\opencv_udemy\contours\contour1.png" 
img=cv2.imread(path) 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
contours,ret1=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
#print(contours) #kontur koordinatları 
cv2.drawContours(img,contours,-1,(0,0,255),3) 
cv2.imshow("contour",img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
