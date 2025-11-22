import cv2
import numpy
import matplotlib
img = cv2.imread(r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg")
"""resimlerin matematiksel değerlerini okur 
ve img değişkenine eşitler"""

#img = cv2.imread(r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg",0)
#img = cv2.imread(r"C:\Users\Amin\opencv_udemy\resimOkumaGostermeKaydetme\indir.png.jpeg", cv2.IMREAD_GRAYSCALE)
"""bu iki satır aynı çıktıyı üretir. cv2.imread fonksiyonunun ikinci argümanı görsele filtre uygular
ve görsel o renkte görünür. opencv'de griye denk gelen sayı 0dır,ya da isteğe göre ikinci şekliyle de yazılabilir."""
#print(img)

cv2.namedWindow("foto", cv2.WINDOW_NORMAL)
"""pencereyi boyutlandırılabilir hale getirmek
için kullanılır"""

cv2.imshow("foto",img)
"""OpenCV’de bir resmi veya videodan 
alınan kareyi ekranda görüntülemek için kullanılır.
ilk argüman açılacak pencerenin adı, ikinci ise
görselin hangi değişkende tutulduğunu gösterir."""
"""cv2.namedWindow ve cv2.imshow fonskiyonlarında pencere
adını belirleyen argümanda ikisi de aynı isimde olmalıdır."""

cv2.imwrite("indir1.jpeg",img)
"""resmi kaydetmek için kullanılır. ilk argüman 
hangi isimle kaydedileceğini, ikincisi ise kaydedilecek görselin
hangi değişkende tutulduğunu gösterir.
pencere açılır ve pencereyi kapattığımız anda  resim çalıştığımız klasöre
girdiğimiz isimle kaydedilir."""
"""resmi python dosyasına değil de başka bir yere kaydetmek istediğimiz zaman 
da ilk argümanda tırnakların arasına kaydedileceği yerin uzantısını yazıyoruz."""

cv2.waitKey(0)
"""girilen değer kadar görseli ekranda tutar
ancak girilen değer milisaniye cinsindendir, yani
1000 yazdığımızda görsel ekranda 1 saniye tutulur
ancak, 0da durum farklıdır, 0 girildiğinde biz 
kapatıncaya kadar görsel ekranda kalır"""
cv2.destroyAllWindows()
"""OpenCV ile açılmış tüm pencereleri 
kapatmak için kullanılır."""
