import cv2 as cv
img = cv.imread('2.png',0)
img = cv.flip(img,1)
bbox = cv.selectROI(img,False)
cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
cv.imwrite('multi_object.png', cut)