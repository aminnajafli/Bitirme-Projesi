import cv2
import numpy as np

canvas=np.zeros((512,512,3),dtype=np.uint8) #512x512'lik bir alanda 
                                            #renkli çizimlerin de yapılabileceği
                                            #siyah tuval oluşturur
a=canvas+255 #siyah tuvali beyaz yapma

print("black canvas", canvas)
print("white canvas:", a)
cv2.imshow("white",a)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
