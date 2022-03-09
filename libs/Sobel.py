import numpy as np
import cv2
from libs.utils import convolution

def sobel(img):
    sobelHKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filteredImg = convolution(img, sobelHKernel)
    filteredImg = convolution(img, sobelVKernel)
    return filteredImg


