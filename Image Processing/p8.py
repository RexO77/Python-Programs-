import cv2
import numpy as np
#Program to rotate, scale and translate an image
image = cv2.imread('images/lena.jpg')
angle = 45
#scaling
scale_x = 1.5
scale_y = 1.5
#translation
tx = 50
ty = 50
#Get the height and width of the image
height, width = image.shape[:2]
#Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()