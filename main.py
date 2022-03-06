from libs import normalization ,equalization ,resizeImg, filters , mean , noise
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load an color image 
img = cv2.imread('images/noisysalterpepper.png') 
#========================================================

# show original image

# cv2.imshow('image',img) # show original image
# cv2.waitKey(0)
#=========================================================

#call normalization
# new_max = 255 
# new_min = 100
# normalization.normalize(img ,new_max ,new_min)
#=========================================================

# call histogram

# equalization.Histogram (img)
# plt.show()
#==========================================================

# call equaliztion

img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img = cv2.convertScaleAbs(img ,alpha= 1.10 ,beta= -20)
equalization.histogram_equaliztion(img)
equaled_hist = equalization.Histogram (img)
filters.average_filter(img, 3)
filters.median_filter(img)
filters.gaussian_filter(img,15,3)
blur = cv2.blur(img,(7,7))
cv2.imwrite("open_cv.png", blur)
# plt.plot(equaled_hist)
# plt.show()
# =====================================================