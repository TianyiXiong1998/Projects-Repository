import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if ret == 0:
        break
    temp = cv.imread("multi_object.png")  # read the template pic
    template = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)  # RGB2Gray
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.namedWindow("template", cv.WINDOW_AUTOSIZE)
    method = cv.TM_CCOEFF_NORMED
    th, tw = template.shape[:2]
    result = cv.matchTemplate(frame, template, method)
    threshold = 0.8
    loc = np.where(result >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(frame, pt, (pt[0] + th, pt[1] + tw), (0, 0, 255), 2)
    cv.imshow("test", frame)
    c = cv.waitKey(10)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()


