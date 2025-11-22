import cv2

cap=cv2.VideoCapture(r"C:\Users\Amin\opencv_udemy\videoOkumaIzleme\o1vid.mp4") 
while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    cv2.imshow("video", frame)
    if cv2.waitKey(30) & 0xFF==ord("q"):
        break

#cv2.waitKey(30) -> her frame 30 ms görünecek ve sonraki frame'e geçilecek


cap.release()
cv2.destroyAllWindows()
