import cv2
import numpy as np
from libs import utils
from random import randint
import numpy as np
from libs import utils


def sp_noise(img):
    salt_and_pepper_image = img.copy()
    row, column = salt_and_pepper_image.shape[:2]
    pixels_no = row * column
    white_pixels = randint(pixels_no // 500, pixels_no // 100)
    black_pixels = randint(pixels_no // 500, pixels_no // 100)

    for i in range(black_pixels):

        y = randint(0, row - 1)
        x = randint(0, column - 1)
        salt_and_pepper_image[y, x] = 0

    for i in range(white_pixels):

        y = randint(0, row - 1)
        x = randint(0, column - 1)
        salt_and_pepper_image[y, x] = 255

    return salt_and_pepper_image


def uniform_noise(img):
    uniform_img = img.copy()
    offset = randint(0, 255)
    # offset = 0
    print(offset)
    row, column = img.shape
    for y in range(0, row):
        for x in range(0, column):

            img[y, x] = (img[y, x] + offset) % 255


    return img


def gauss_noise(img):
        mean = utils.mean_grey(img)
        std = utils.std_grey(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row , column = img.shape
        gauss = np.random.normal(mean,std,(row,column))
        gauss = np.asanyarray(gauss , dtype= np.uint8)
        noisy = img + gauss                                                      
        return noisy