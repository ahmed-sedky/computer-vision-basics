import cv2
import numpy as np



def image_mean(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row , column = img.shape
        sum = 0
        for y in range(0,row):
                for x in range(0,column):
                       
                       sum = sum + img[y,x] 

        img_mean = sum / img.size
        print(img_mean)
        return img_mean            
        #print(cv2.mean(img)[0])   

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
        print(std)
        #print(cv2.meanStdDev(img))
        return std