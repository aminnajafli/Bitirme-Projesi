import cv2
import numpy as np

img = cv2.imread("road.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(edges,rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  


cv2.imshow("edges", edges)
cv2.imshow("Hough Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
