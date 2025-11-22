import cv2
import numpy as np 

path1=r"C:\Users\Amin\opencv_udemy\coreOperations\bitwise_1.png"
img1=cv2.imread(path1)
path2=r"C:\Users\Amin\opencv_udemy\coreOperations\bitwise_2.png"
img2=cv2.imread(path2)

bitAnd=cv2.bitwise_and(img2,img1)
#resimleri karşılaştırarak "and" mantıksal 
#operatörünün yaptığı gibi bir birleşme yapar

bitOr=cv2.bitwise_or(img2,img1)
#resimleri karşılaştırarak "or" mantıksal 
#operatörünün yaptığı gibi bir birleşme yapar

bitXor=cv2.bitwise_xor(img2,img1)
#resimleri karşılaştırarak "or" mantıksal 
#operatörünün yaptığı gibi bir birleşme yapar

bitNot1=cv2.bitwise_not(img1)
bitNot2=cv2.bitwise_not(img2)
#resimleri karşılaştırarak "or" mantıksal 
#operatörü mantığıyla siyah yerleri beyaz, beyaz yerleri siyah yapar

#bu işlemlerde siyah yerler 0, beyaz yerler 1 bit olarak algılanır 

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img3",bitAnd)
cv2.imshow("img4",bitOr)
cv2.imshow("img5",bitXor)
cv2.imshow("img6",bitNot1)
cv2.imshow("img7",bitNot2)


cv2.waitKey(0)
cv2.destroyAllWindows()

