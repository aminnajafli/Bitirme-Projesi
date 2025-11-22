import cv2
import numpy as np
import matplotlib.pyplot as plt
#videoda beyaz bir köpek var ve bu köpeği 
#video boyunca takip edeceğiz
#diğer nesneleri görmeden


cap=cv2.VideoCapture(r"C:\Users\Amin\opencv_udemy\contours\dog.mp4")
while(1):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    sensitivity=15
    lower_white=np.array([0,0,255-sensitivity])
    upper_white=np.array([255,sensitivity,255])
    #köpeğin renklerine uygun renkler alınıyor
    #bu renk aralıkları google'dan bulunabilir
    #hsv code for ... color
    mask=cv2.inRange(hsv,lower_white,upper_white)
    #upper_white ve lower_white aralığındaki nesnelere maske uygulanması
    #geri kalan kısmın da silinmesini sağlar
    res=cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("result",res)

    k=cv2.waitKey(5)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()
