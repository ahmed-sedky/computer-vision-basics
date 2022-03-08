from libs import normalization ,equalization ,resizeImg, filters , noise, graphs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Load an color image 
img = Image.open("images/lenna.png") 
#========================================================

# show original image3

# cv2.imshow('image',img) # show original image
# cv2.waitKey(0)
#=========================================================

#call normalization
# new_max = 255 
# new_min = 100
# normalization.normalize(img ,new_max ,new_min)
#=========================================================

# call histogram
# graphs.histogram (img)
#==========================================================

# call equaliztion and filter

img = img.convert('L')
img.show()
#img = cv2.convertScaleAbs(img ,alpha= 1.10 ,beta= -20)
# equalization.Histogram(img)
# plt.show()
# equalization.histogram_equaliztion(img)
# equaled_hist = equalization.Histogram (img)
# filters.average_filter(img, 3)
#filters.median_filter(img)
# filters.gaussian_filter(img,3,3)
# blur = cv2.blur(img,(7,7))
# cv2.imwrite("open_cv.png", blur)
# plt.plot(equaled_hist)
# plt.show()
# =====================================================

# call distribution curve
# graphs.distribution_curve(img)