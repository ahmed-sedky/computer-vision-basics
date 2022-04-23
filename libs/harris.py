import numpy as np
import cv2
import matplotlib.pyplot as plt
from libs import filters,Sobel
def harrisCorner(image ,k):
    grayscale_image = filters.grayscale(image)
    Ix,Iy = Sobel.sobel(image,True)
    Ixx = filters.gaussian_filter(Ix**2, size = 5, std_dev=  1)
    Iyy = filters.gaussian_filter(Iy**2, size = 5, std_dev=  1)
    Ixy = filters.gaussian_filter(Ix*Iy, size = 5, std_dev=  1)

    det = Ixx *Iyy - (Ixy**2)
    trace = Ixx + Iyy
    harrisResponse = det - k * (trace **2)

    return harrisResponse

def corner2Image (image, harrisResponse , cornerThreshold = 0.01 ):
    cop_harris = np.copy(harrisResponse)
    harrisMatrix =cv2.dilate(cop_harris,None)
    hMax  = harrisMatrix.max()
    hMin = harrisMatrix.min()
    corner_indices = np.array(harrisMatrix > (hMax * cornerThreshold), dtype="int8")
    src = np.copy(image)
    corner_indices = corner_indices[2:,2:]
    src[corner_indices == 1 ] = [0 , 255 ,0]
    return src    

