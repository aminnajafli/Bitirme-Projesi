import cv2
import numpy as np
import matplotlib.pyplot as plt
#cv2.Canny(x,y,z)
#x-kenarlarını arayacağımız görsel
#y ve z-resmin threshold aralıkları

cap=cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)

    edges=cv2.Canny(frame,100,200)
    #genelde varsayılan olarak 100 ve 200 girilir
    cv2.imshow("frame",frame)
    cv2.imshow("edges",edges)

    if cv2.waitKey(5)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
