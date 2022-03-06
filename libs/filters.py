from ctypes import sizeof
from math import ceil, floor, pi
from cv2 import exp
import numpy as np
import cv2
from rsa import sign
from scipy import  signal

def create_gaussian_kernel (size:int, std_dev:float):
    kernel = np.fromfunction(lambda x, y: (1/(2*pi*std_dev**2)) * exp((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*std_dev**2)),
     (size, size))
    return kernel/np.sum(kernel)


def median (array):
    sorted_array = sorted(array)
    size= len(sorted_array)
    if (size % 2) == 0:
        median = (sorted_array[ceil(size/2)] + sorted_array[floor(size/2)]) /2
    else:
        median = sorted_array[floor(size/2)]
    return median


def average_filter(image, kernel_size:int):
    kernel = np.ones([kernel_size,kernel_size], dtype=int)
    kernel = kernel/(pow(kernel_size,2))
    
    filtered_image = signal.convolve2d(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    cv2.imwrite('average_filtered_image.png', filtered_image)

def median_filter(image):
    rows, cols = image.shape
    filtered_image = np.zeros([rows,cols])
    for i in range (1, rows-1):
        for j in range(1, cols-1):
            kernel = [image[i-1, j-1], image[i-1, j], image[i-1, j + 1], image[i, j-1],
               image[i, j],
               image[i, j + 1],
               image[i + 1, j-1],
               image[i + 1, j],
               image[i + 1, j + 1]]
            filtered_image[i,j] = median(kernel)
    filtered_image = filtered_image.astype(np.uint8)
    cv2.imwrite('median_filtered_image.png', filtered_image)

def gaussian_filter(image, size:int, std_dev:float):
    kernel  = create_gaussian_kernel(size,std_dev)
    filtered_image = signal.convolve2d(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    cv2.imwrite('gaussian_filtered_image.png', filtered_image)

    

