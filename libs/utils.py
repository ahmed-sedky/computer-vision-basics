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


def mean_grey(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row , column = img.shape
        sum = 0
        for y in range(0,row):
                for x in range(0,column):               
                       sum = sum + img[y,x] 

        img_mean = sum / img.size
        #print(img_mean)
        return img_mean            
        #print(cv2.mean(img)[0])   




def std_grey(img):
        m = mean_grey(img) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row , column = img.shape
        sum = 0
        for y in range(0,row):
                for x in range(0,column):
                       z = (img[y,x] - m) ** 2
                       sum = sum + z
        std = (sum / img.size)**0.5
        #print(std)
        #print(cv2.meanStdDev(img))
        return std

        


def mean_rgb(img):
        row, column = img.shape[:2]
        sum_blue = sum_green = sum_red = 0
        size = row * column
        for y in range(0,row):
                for x in range(0,column):               
                       sum_blue = sum_blue + img[y,x][0]
                       sum_green = sum_green + img[y,x][1]
                       sum_red = sum_red + img[y,x][2] 

        img_mean = [sum_blue/ size , sum_green/size , sum_red/size  ]
        print(img_mean)
        print(cv2.mean(img))
        #print(cv2.meanStdDev(img))
        return img_mean





def std_rgb(img):
        m = mean_rgb(img) 
        row , column = img.shape[:2]
        size = row * column
        sum_blue = sum_red = sum_green = 0
        for y in range(0,row):
                for x in range(0,column):
                       sum_blue = sum_blue + (img[y,x][0] - m[0]) ** 2
                       sum_green = sum_green + (img[y,x][1] - m[1]) ** 2
                       sum_red = sum_red + (img[y,x][2] - m[2]) ** 2
                
        std_blue = (sum_blue / size)**0.5
        std_red = (sum_red / size)**0.5
        std_green = (sum_green / size)**0.5
        std = [ std_blue , std_green , std_red]
        print(std)
        print(cv2.meanStdDev(img))
        return std