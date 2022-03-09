import numpy as np
import cv2
from libs.utils import convolution

def roberts(img):

    robertsKernel = np.array([[1, 0], [0, -1]])
    filteredImg = convolution(img, robertsKernel)
    filteredImg = convolution(img, robertsKernel)
    return filteredImg

