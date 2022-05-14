import numpy as np
from libs.utils import convolution

def sobel(img):
    sobelHKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVKernel = np.flip(sobelHKernel.T)
    filteredImgH = convolution(img, sobelHKernel)
    filteredImgV = convolution(img, sobelVKernel)
    return np.sqrt(pow(filteredImgH, 2.0) + pow(filteredImgV, 2.0))
   

