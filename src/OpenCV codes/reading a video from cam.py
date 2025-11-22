import cv2

cap=cv2.VideoCapture(0) #cv2.imread() ile aynı sadece bu fonksiyonla 
                        #bilgisayardan da görüntü eklenebilir.
                        #bilgisayardan bir video girilecekse, parantez içine adresi yazılır, bilgisayar
                        #kamerası kullanılacaksa, 0, başka kameralar kullanılacaksa 1...6
while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

#cv2.waitKey(30) -> her frame 30 ms görünecek ve sonraki frame'e geçilecek


cap.release()
cv2.destroyAllWindows()
