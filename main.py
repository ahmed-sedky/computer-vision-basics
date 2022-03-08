from libs import normalization ,equalization ,resizeImg, filters , mean , noise
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load an color image 
img = cv2.imread('images/lenna.png') 
#========================================================

#call filters
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# filters.average_filter(img, 3)
# filters.median_filter(img)
# filters.gaussian_filter(img,15,3)
# blur = cv2.blur(img,(7,7))
# cv2.imwrite("open_cv.png", blur)
#=================================================================

# call histogram
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# equalization.Histogram (img)
# plt.show()
#==========================================================

# call equaliztion

img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
equalization.histogram_equaliztion(img)
equaled_hist = equalization.Histogram (img)
plt.show()
# =====================================================

#call normalization
# new_max = 255 
# new_min = 100
# normalization.normalize(img ,new_max ,new_min)
#=========================================================

# show original image

cv2.imshow('image',img) # show original image
cv2.waitKey(0)
#=========================================================

