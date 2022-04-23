import numpy as np
import cv2
from libs.utils import convolution
from scipy.signal import convolve2d
def sobel(img,harris):
    # sobelHKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobelVKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # filteredImg = convolution(img, sobelHKernel)
    # filteredImg = convolution(img, sobelVKernel)
    # return filteredImg
    # convolution
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    horizontal_edge = convolve2d(gray, horizontal)
    vertical_edge = convolve2d(gray, vertical)
    if(harris == True):
        return horizontal_edge,vertical_edge
    else:
        mag = np.sqrt(pow(horizontal_edge, 2.0) + pow(vertical_edge, 2.0))
        return mag
    


