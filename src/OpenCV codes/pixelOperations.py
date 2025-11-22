import cv2
import numpy as np

path=r"C:\Users\Amin\opencv_udemy\coreOperations\testImages\opencv.png"
img=cv2.imread(path)
#print(img)

px=img[35,0]
print(px) #görüntüde istenilen koordinattaki
          #pikselleri görmemizi sağlar


# accessing pixel's value_1 BGR -> 012
blue=img[100,100,0]
green=img[100,100,1]
red=img[100,100,2] #(100,100) koordinatındakı red değeri

# accessing pixel's value_2
img.item(10,10,2)

# manipulations on pixels
img[100,100,0]=201
img[100,100,1]=117
img[100,100,2]=44





