import cv2
import numpy as np 
import matplotlib.pyplot as plt
#görselin köşeleri bulmak
img=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\text.png")
img1=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\contour.png")

#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

gray=np.float32(gray)
corners=cv2.goodFeaturesToTrack(gray,50,0.01,10)
#parametreler
#1. kenarları bulunacak görselin adı
#2. max köşe sayısı
#3. quality label (kalite etiketi)-fix olarak 0.01'dir
#4. köşeler arasındaki min mesafe
corners=np.intc(corners) #int8 değil, intc aksi halde overflow olur
#işlem yapabilmek için öncelikle gray'i float32 tipine
#köşeleri çizerken de float değer kullanamadığımız için 
#corners'i int boyutuna çeviriyoruz
for corner in corners:
    x,y=corner.ravel()
    #cv2.circle(img,(x,y),3,(0,0,255),-1)
    cv2.circle(img1,(x,y),3,(0,0,255),-1)

#cv2.imshow("corner",img)
cv2.imshow("corner",img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
