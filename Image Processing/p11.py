#image Countour
import cv2
import numpy as np
image = cv2.imread('images/lena.jpg')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(grey,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)
cnt = contours[0]
cv2.drawContours(image,[cnt],0,(0,255,0),3)
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()