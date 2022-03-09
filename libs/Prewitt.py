import numpy as np
import cv2
from libs.utils import convolution

def prewitt(img):
    prewittHKernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittVKernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    filteredImg = convolution(img, prewittHKernel)
    filteredImg = convolution(img, prewittVKernel)
    return filteredImg
