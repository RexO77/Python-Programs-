import cv2 #image processing library
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
image = cv2.imread('images/lena.jpg') #read the image
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grey,1,1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle on the image
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()