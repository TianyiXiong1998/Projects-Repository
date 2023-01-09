import cv2 as cv
import numpy as np
img = cv.imread('1.png',0)
img = cv.flip(img,1)
bbox = cv.selectROI(img,False)
cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
cv.imwrite('single_object.png', cut)
capture = cv.VideoCapture(0)  # 打开摄像头
while True:
    ret, frame = capture.read()
    if ret == 0:
        break
    temp = cv.imread("single_object.png")  # read the template pic
    template = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)  # RGB2Gray
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.namedWindow("template", cv.WINDOW_AUTOSIZE)
    method = cv.TM_SQDIFF_NORMED
    th, tw = template.shape[:2]
    result = cv.matchTemplate(frame, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    print("max:", max_val)
    print("min:", min_val)
    if method == cv.TM_SQDIFF_NORMED:
        tl = min_loc
    else:
        tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    if min_val <= 0.15:
        if max_val <= 1:
            cv.rectangle(frame, tl, br, (0, 0, 255), 2)
            cv.putText(frame, "Detected!!!", (0, 60), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 2)

    else:
        cv.putText(frame, "Not detected", (0, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 255, 0), 1)
    cv.imshow("test", frame)
    c = cv.waitKey(10)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.waitKey(0)
cv.destroyAllWindows()
