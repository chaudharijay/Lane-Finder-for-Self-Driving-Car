import cv2
import numpy as np
import matplotlib.pyplot as plt

#Loading Image
#reads image and return numpy array for each pixel
#image = cv2.imread('test_image.jpg') 

# cv2.imshow('result', image) 

""" this solo wont show image"""

# cv2.waitKey(0) 

""" shows iimage for a specified amount of time """

# step 1: Convert image to grayscale

#Edge detection - identifying sharp changes in intensity in adjacent pixels [0 255] --->[black white]
# we will convert our image into grayscale because it has only 1 channel and colored images have 3 channels so its easy to compute

# lane_image = np.copy(image)
# gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
# cv2.imshow("result", gray)
# cv2.waitKey(0)

# step 2 : Reduce Noise using gaussian blurr

# blur = cv2.GaussianBlur(gray, (5,5), 0)
# cv2.imshow("result", blur)
# cv2.waitKey(0)

# step 3 : Finding Edges

# using canny function we will check the gradient to check sudden change in gradient stronger the better

# canny = cv2.Canny(blur, 50, 150)
# cv2.imshow("result", canny)
# cv2.waitKey(0)

# step 4: Finding Lane lines

def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200, )])

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

plt.imshow(canny)
plt.show()

