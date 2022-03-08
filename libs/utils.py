import imp
import numpy as np
import cv2
from math import ceil, floor


def rgb_to_bw(pixel):
    return (
        round(0.2126 * pixel[2]
        + 0.7152 * pixel[1]
        + 0.0722 * pixel[0])
    )


def median(array):
    sorted_array = sorted(array)
    size = len(sorted_array)
    if (size % 2) == 0:
        median = (sorted_array[ceil(size / 2)] + sorted_array[floor(size / 2)]) / 2
    else:
        median = sorted_array[floor(size / 2)]
    return median

def image_mean(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row , column = img.shape
    sum = 0
    for y in range(0,row):
            for x in range(0,column):
                    sum = sum + img[y,x] 
    img_mean = sum / img.size   
    return img_mean            

#img = cv2.imread("C:/Users/Mo/Desktop/CV/apple.jpeg")
#image_mean(img) 


def image_standard_deviation(img):
    m = image_mean(img) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row , column = img.shape
    sum = 0
    for y in range(0,row):
            for x in range(0,column):
                    z = (img[y,x] - m) ** 2
                    sum = sum + z
    std = (sum / img.size)**0.5
    return std