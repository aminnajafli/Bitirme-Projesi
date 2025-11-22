import cv2

cv2.namedWindow("klon")
img = cv2.imread(r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg")

img=cv2.resize(img,(1000,480))
"""açılan pencereyi yeniden boyutlandırmayı sağlar"""

cv2.imshow("klon", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
