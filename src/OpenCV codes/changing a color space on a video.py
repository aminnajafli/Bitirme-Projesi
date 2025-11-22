import cv2
import numpy as np 

path=r"C:\Users\Amin\opencv_udemy\catty.mp4"
vid=cv2.VideoCapture(path)

while True:
    ret, frame=vid.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        break

    cv2.imshow("catty", frame)
    if cv2.waitKey(30) & 0xFF==ord("q"):
        break


vid.release()
cv2.destroyAllWindows()
