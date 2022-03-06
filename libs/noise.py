from cv2 import imshow
import numpy as np
import cv2
from random import randint

def sp_noise(img):

 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 row , column = img.shape
 pixels_no = row * column
 white_pixels = randint(pixels_no//500, pixels_no//100)
 black_pixels = randint(pixels_no//500, pixels_no//100)

 for i in range(black_pixels): 

         y=randint(0, row - 1)
         x=randint(0, column - 1)
         img[y,x] = 0

 for i in range(white_pixels):
       
         y=randint(0, row - 1)
         x=randint(0, column - 1)
         img[y,x] = 255

 return img




def uniform_noise(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        offset = randint(0,255)
        #offset = 0
        print(offset)
        row , column = img.shape
        #row = img.shape[0]
        #column = img.shape[1]
        for y in range(0,row):
                for x in range(0,column):
                       
                       img[y,x] = (img[y,x] + offset) % 255
                       #img[y,x][1] = (img[y,x][1] + offset )% 255
                       #img[y,x][2] = (img[y,x][2] + offset) % 255
                      
        
        return img



# def gauss_noise(img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gauss = np.random.normal(mean,sigma,(row,col,ch))
#         gauss = gauss.reshape(row,col,ch)
#         noisy = img + gauss