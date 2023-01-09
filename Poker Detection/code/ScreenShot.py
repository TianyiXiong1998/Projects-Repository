import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#
img_list = ["1.png","2.png"]
for name in img_list:
    capture = cv2.VideoCapture(0)
    while (True):
        # get a frame
        ret, frame = capture.read()
        frame = cv2.flip(frame,1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frameqqq
        cv2.imshow("capture", frame)  # 生成摄像头窗口

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
            cv2.imwrite(name, frame)  # 保存路径
            break


capture.release()
cv2.destroyAllWindows()