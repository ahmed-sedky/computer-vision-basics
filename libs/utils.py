import cv2
import numpy as np
from math import ceil, floor

def convolution(image, kernel):
    convolvedImage = np.zeros(image.shape)
    paddingHeight = int((len(kernel) - 1) / 2)
    paddingWidth = int((len(kernel[0]) - 1) / 2)

    padded_image = np.zeros(
        (len(image) + (2 * paddingHeight), len(image[0]) + (2 * paddingWidth))
    )

    padded_image[
        paddingHeight : padded_image.shape[0] - paddingHeight,
        paddingWidth : padded_image.shape[1] - paddingWidth,
    ] = image
    for row in range(len(image)):
        for col in range(len(image[0])):
            convolvedImage[row, col] = np.sum(
                kernel
                * padded_image[row : row + len(kernel), col : col + len(kernel[0])]
            )
    return convolvedImage

def rgb_to_bw(pixel):
    return round(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])


def median(array):
    sorted_array = sorted(array)
    size = len(sorted_array)
    if (size % 2) == 0:
        median = (sorted_array[ceil(size / 2)] + sorted_array[floor(size / 2)]) / 2
    else:
        median = sorted_array[floor(size / 2)]
    return median


def max_min(img):
    if img.mode == 'L':
        min_gray,max_gray = max_min_gray(img)
        return min_gray,max_gray
    else : 
        min_b,max_b,min_g,max_g,min_r,max_r = max_min_RGB(img)
        return min_b,max_b,min_g,max_g,min_r,max_r

def max_min_RGB(img):
    rows,cols = img.size 
    min_b = 1000 ; max_b = 0 ; min_g =1000 ; max_g =0 ; min_r =1000 ; max_r =0
    pixels = img.load()
    for i in range(rows):
        for j in range(cols):
            if pixels[i,j][2] < min_r:
                min_r = pixels[i,j][2]
            if pixels[i,j][1] < min_g:
                min_g = pixels[i,j][1]
            if pixels[i,j][0] < min_b:
                min_b = pixels[i,j][0]

            if pixels[i,j][2] > max_r:
                max_r = pixels[i,j][2]
            if pixels[i,j][1] > max_g:
                max_g = pixels[i,j][1]
            if pixels[i,j][0] > max_b:
                max_b = pixels[i,j][0]
    return min_b,max_b,min_g,max_g,min_r,max_r

def max_min_gray(img):
    rows, columns =img.size
    min_gray = 1000 ; max_gray = 0 
    pixels = img.load()
    for i in range(rows):
        for j in range(columns):
            if pixels[i,j] < min_gray:
                min_gray = pixels[i,j]
            if pixels[i,j] > max_gray:
                max_gray = pixels[i,j]
    return min_gray,max_gray


def image_mean(image):
    if (len(image.shape)<2):
        return mean_grey(image)
    elif (len(image.shape)==3):
        return mean_rgb(image)


def mean_grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, column = img.shape
    sum = 0
    for y in range(0, row):
        for x in range(0, column):
            sum = sum + img[y, x]

    img_mean = sum / img.size
    return img_mean


def mean_rgb(img):
    row, column = img.shape[:2]
    sum_blue = sum_green = sum_red = 0
    size = row * column
    for y in range(0, row):
        for x in range(0, column):
            sum_blue = sum_blue + img[y, x][0]
            sum_green = sum_green + img[y, x][1]
            sum_red = sum_red + img[y, x][2]

    img_mean = [sum_blue / size, sum_green / size, sum_red / size]
    return img_mean


def image_standard_deviation(image):
    if (len(image.shape)<2):
        return std_grey(image)
    elif (len(image.shape)==3):
        return std_rgb(image)


def std_grey(img):
    m = mean_grey(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, column = img.shape
    sum = 0
    for y in range(0, row):
        for x in range(0, column):
            z = (img.getpixel((y, x)) - m) ** 2
            sum = sum + z
    std = (sum / img.size) ** 0.5
    return std


def std_rgb(img):
    m = mean_rgb(img)
    row, column = img.shape[:2]
    size = row * column
    sum_blue = sum_red = sum_green = 0
    for y in range(0, row):
        for x in range(0, column):
            sum_blue = sum_blue + (img[y, x][0] - m[0]) ** 2
            sum_green = sum_green + (img[y, x][1] - m[1]) ** 2
            sum_red = sum_red + (img[y, x][2] - m[2]) ** 2

    std_blue = (sum_blue / size) ** 0.5
    std_red = (sum_red / size) ** 0.5
    std_green = (sum_green / size) ** 0.5
    std = [std_blue, std_green, std_red]
    return std


def get_pixel_values(image):
    if (len(image.shape)<2):
        return grayscale_values(image)
    elif (len(image.shape)==3):
        return rgb_values(image)


def grayscale_values(image):
    height, width = image.size
    pixel_values = []
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image[i, j][k]
                pixel_values.append(pixel)
    return pixel_values


def rgb_values(image):
    height, width = image.shape[:2]
    rgb_values = [[], [], []]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image[i, j][k]
                rgb_values[k].append(pixel)
    return rgb_values


def frequencies_of_pixel_values(image):
    if (len(image.shape)<2):
        return grayscale_frequencies(image)
    elif (len(image.shape)==3):
        return rgb_frequencies(image)


def grayscale_frequencies(image):
    frequencies = {}
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel = image.getpixel((i, j))
            frequencies[pixel] = frequencies[pixel] + 1 if (pixel in frequencies) else 1
    return frequencies


def rgb_frequencies(image):
    frequencies = [{}, {}, {}]
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image[i, j][k]
                frequencies[k][pixel] = (
                    frequencies[k][pixel] + 1 if (pixel in frequencies[k]) else 1
                )
    return frequencies

def convert_RGB_to_grayscale(image):
   width, height = image.size
   pixels = image.load()
   for i in range(width):
       for j in range(height):
           pixels[i, j] = (rgb_to_bw(pixels[i, j]), rgb_to_bw(pixels[i, j]), rgb_to_bw(pixels[i, j])) 