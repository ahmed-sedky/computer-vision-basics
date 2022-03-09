import cv2
from math import ceil, floor


def rgb_to_bw(pixel):
    return round(0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0])


def median(array):
    sorted_array = sorted(array)
    size = len(sorted_array)
    if (size % 2) == 0:
        median = (sorted_array[ceil(size / 2)] + sorted_array[floor(size / 2)]) / 2
    else:
        median = sorted_array[floor(size / 2)]
    return median


def max_min(img):
    if len(img.shape) == 3:
        min_b,max_b,min_g,max_g,min_r,max_r = max_min_RGB(img)
        return min_b,max_b,min_g,max_g,min_r,max_r
    else : 
        min_gray,max_gray = max_min_gray(img)
        return min_gray,max_gray

def max_min_RGB(img):
    rows,cols,_ = img.shape 
    min_b = 1000 ; max_b = 0 ; min_g =1000 ; max_g =0 ; min_r =1000 ; max_r =0
    for i in range(rows):
        for j in range(cols):
            if img[i,j][2] < min_r:
                min_r = img[i,j][2]
            if img[i,j][1] < min_g:
                min_g = img[i,j][1]
            if img[i,j][0] < min_b:
                min_b = img[i,j][0]

            if img[i,j][2] > max_r:
                max_r = img[i,j][2]
            if img[i,j][1] > max_g:
                max_g = img[i,j][1]
            if img[i,j][0] > max_b:
                max_b = img[i,j][0]
    return min_b,max_b,min_g,max_g,min_r,max_r

def max_min_gray(img):
    shape =img.shape
    min_gray = 1000 ; max_gray = 0 
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i,j] < min_gray:
                min_gray = img[i,j]
            if img[i,j] > max_gray:
                max_gray = img[i,j]
    return min_gray,max_gray


def image_mean(image):
    if image.mode == "L":
        return mean_grey(image)
    elif image.mode == "RGB":
        return mean_rgb(image)


def mean_grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, column = img.size
    sum = 0
    for y in range(0, row):
        for x in range(0, column):
            sum = sum + img.getpixel((y, x))

    img_mean = sum / img.size
    return img_mean


def mean_rgb(img):
    row, column = img.size
    sum_blue = sum_green = sum_red = 0
    size = row * column
    for y in range(0, row):
        for x in range(0, column):
            sum_blue = sum_blue + img.getpixel((y, x))[0]
            sum_green = sum_green + img.getpixel((y, x))[1]
            sum_red = sum_red + img.getpixel((y, x))[2]

    img_mean = [sum_blue / size, sum_green / size, sum_red / size]
    return img_mean


def image_standard_deviation(image):
    if image.mode == "L":
        return std_grey(image)
    elif image.mode == "RGB":
        return std_rgb(image)


def std_grey(img):
    m = mean_grey(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, column = img.size
    sum = 0
    for y in range(0, row):
        for x in range(0, column):
            z = (img.getpixel((y, x)) - m) ** 2
            sum = sum + z
    std = (sum / img.size) ** 0.5
    return std


def std_rgb(img):
    m = mean_rgb(img)
    row, column = img.size
    size = row * column
    sum_blue = sum_red = sum_green = 0
    for y in range(0, row):
        for x in range(0, column):
            sum_blue = sum_blue + (img.getpixel((y, x))[0] - m[0]) ** 2
            sum_green = sum_green + (img.getpixel((y, x))[1] - m[1]) ** 2
            sum_red = sum_red + (img.getpixel((y, x))[2] - m[2]) ** 2

    std_blue = (sum_blue / size) ** 0.5
    std_red = (sum_red / size) ** 0.5
    std_green = (sum_green / size) ** 0.5
    std = [std_blue, std_green, std_red]
    return std


def get_pixel_values(image):
    if image.mode == "L":
        return grayscale_values(image)
    elif image.mode == "RGB":
        return rgb_values(image)


def grayscale_values(image):
    height, width = image.size
    pixel_values = []
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image.getpixel((i, j))[k]
                pixel_values.append(pixel)
    return pixel_values


def rgb_values(image):
    height, width = image.size
    rgb_values = [[], [], []]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image.getpixel((i, j))[k]
                rgb_values[k].append(pixel)
    return rgb_values


def frequencies_of_pixel_values(image):
    if image.mode == "L":
        return grayscale_frequencies(image)
    elif image.mode == "RGB":
        return rgb_frequencies(image)


def grayscale_frequencies(image):
    frequencies = {}
    height, width = image.size
    for i in range(height):
        for j in range(width):
            pixel = image.getpixel((i, j))
            frequencies[pixel] = frequencies[pixel] + 1 if (pixel in frequencies) else 1
    return frequencies


def rgb_frequencies(image):
    frequencies = [{}, {}, {}]
    height, width = image.size
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image.getpixel((i, j))[k]
                frequencies[k][pixel] = (
                    frequencies[k][pixel] + 1 if (pixel in frequencies[k]) else 1
                )
    return frequencies
