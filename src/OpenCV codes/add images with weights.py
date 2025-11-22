import cv2
import numpy as np
#resimlerde toplama işlemi yapabilmek için 
#resimlerin boyutlarının aynı olması gerekmektedir
#çünkü resimler sayısal olarak bir matris boyutundadır
#ve matrislerde toplama işlemi yapabilmek için 
#matris boyutlarının aynı olması gerekir.
circle=np.zeros((512,512,3),np.uint8)+255
cv2.circle(circle,(256,256),60,(0,255,0),-1)

rectangle=np.zeros((512,512,3),np.uint8)+255
cv2.rectangle(rectangle,(150,150),(350,350),(0,0,255),-1)

dst=cv2.addWeighted(circle,0.6,rectangle,0.4,0)
#iki bilinmeyenli denklem olarak düşünülebilir
#f(x,y)=ax+by+c
#burda a=0.6, b=0.4, c=0, x=circle, y=rectangle
#a+b=1 olmak zorunda her zaman


cv2.imshow("circle",circle)
cv2.imshow("rectangle",rectangle)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
