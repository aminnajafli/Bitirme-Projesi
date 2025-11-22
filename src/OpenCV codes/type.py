import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas=np.zeros((512,512,3),dtype=np.uint8)+255
canvas1=np.zeros((512,512,3),dtype=np.uint8)+255
canvas2=np.zeros((512,512,3),dtype=np.uint8)+255

font1=cv2.FONT_HERSHEY_COMPLEX
font2=cv2.FONT_HERSHEY_SIMPLEX
font3=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(canvas,"openCV",(10,100),font1,4,(0,0,0),cv2.LINE_AA)
cv2.putText(canvas1,"openCV",(10,100),font2,4,(0,0,0),cv2.LINE_AA)
cv2.putText(canvas2,"openCV",(10,100),font3,4,(0,0,0),cv2.LINE_AA)
#yazı eklemek için kullanılan fonksiyondur
#parametreleri:
#1. yazı yazılacak tuvalin adı
#2. yazılacak yazı
#3. yazının başlangıç koordinatı
#4. yazıda kullanılacak font
#5. yazının rengi
#6. yazı tipi


cv2.imshow("canvas",canvas)
cv2.imshow("canvas1",canvas1)
cv2.imshow("canvas2",canvas2)
cv2.waitKey(0)
cv2.destroyAllWindows()
