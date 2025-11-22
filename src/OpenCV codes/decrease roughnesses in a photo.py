#görseldeki gürültüleri (uyumsuz değerler,outliers) azaltma kodu
import cv2
import numpy as np

img_filter=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\duzltmopc.png")
img_bozuk=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\bozukopc.png")
img_bt=cv2.imread(r"C:\Users\Amin\opencv_udemy\coreOperations\bilateral.png")

blur=cv2.blur(img_filter,(15,15)) #resmin yumşama değeridir
#sadece pozitif tek sayılar yazılabilir.
#değerleri oynatarak farklı seviyede blurluklar uygulayabiliyoruz

blur2=cv2.GaussianBlur(img_filter,(15,15),cv2.BORDER_DEFAULT)
#GaussianBlur metodunda sonda sınır çizgileri ile ilgili parametre var
#oraya varsayılan neyse o şekilde ayarlanması için BORDER_DEFAULT girilir

blur_m=cv2.medianBlur(img_bozuk,15)

blur_b=cv2.bilateralFilter(img_bt,9,95,75)

cv2.imshow("origbt",img_bt)
cv2.imshow("original",img_filter)
cv2.imshow("blur",blur)
cv2.imshow("blur2",blur2)
cv2.imshow("blurmed",blur_m)
cv2.imshow("blurbt",blur_b)
cv2.waitKey(0)
cv2.destroyAllWindows()
