import cv2
import numpy as np
import matplotlib.pyplot as plt

path=r"C:\Users\Amin\opencv_udemy\coreOperations\testImages\cat-1543541_640.jpg"
img=cv2.imread(path)
#print(img)
corner=img[150:300, 0:100] #(y_start:y_end,x_start:x_end) e göre tarama yapılır
#cv2.imshow("corner",corner)

img[150:300,100:290]=(117,44,138) 
#girilen korrdinatlar arasındaki kısım girilen renge eşit olacak
#ve çıktıda da resmin o kısmı değişmiş olacak

cv2.imshow("cat", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
