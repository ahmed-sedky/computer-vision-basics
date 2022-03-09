from libs import normalization, equalization, resizeImg, filters, noise, graphs, utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load an color image
# img = Image.open("images/img2.png")
# ========================================================

# show original image3
# call filters
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# filters.average_filter(img, 3)
# filters.median_filter(img)
# filters.gaussian_filter(img,15,3)
# blur = cv2.blur(img,(7,7))
# cv2.imwrite("open_cv.png", blur)
# =================================================================

# call histogram
# graphs.histogram (img)
# ==========================================================

# call equaliztion and filter

# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# img = img.convert('L')
img = cv2.imread('images/lenna.png') 
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
equalization.equalize(img)
equaled_hist = equalization.Histogram (img)
plt.show()
cv2.imshow('image',img) # show original image
cv2.waitKey(0)
# img = img.convert('L')
# img.show()
# =====================================================

# call distribution curve
# graphs.distribution_curve(img)

#call normalization
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# img = img.convert('L')
# utils.convert_RGB_to_grayscale(img)
# new_max = 255
# new_min = 100
# normalization.normalize(img ,new_max ,new_min)
#=========================================================

# show original image
# img.show()

# cv2.imshow('image',img) # show original image
# cv2.waitKey(0)
# =========================================================
