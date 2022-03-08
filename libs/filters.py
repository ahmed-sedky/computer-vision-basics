from ctypes import sizeof
from math import ceil, floor, pi, exp
from matplotlib import pyplot as plt
import numpy as np 
from rsa import sign
from scipy import signal
from libs.utils import median
import PIL


def create_gaussian_kernel(kernel_size: int, std_dev: float):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * pi * std_dev ** 2))
        * exp(
            (-1 * ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
            / (2 * std_dev ** 2)
        ),
        (kernel_size, kernel_size),
    )
    return kernel / np.sum(kernel)





def average_filter(image, kernel_size: int):
    kernel = np.ones([kernel_size, kernel_size], dtype=int)
    kernel = kernel / (pow(kernel_size, 2))

    filtered_image = signal.convolve2d(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    image.show(filtered_image)


def median_filter(image):
    rows, cols = image.size
    filtered_image = np.zeros([rows, cols])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            kernel = [
                image[i - 1, j - 1],
                image[i - 1, j],
                image[i - 1, j + 1],
                image[i, j - 1],
                image[i, j],
                image[i, j + 1],
                image[i + 1, j - 1],
                image[i + 1, j],
                image[i + 1, j + 1],
            ]
            filtered_image[i, j] = median(kernel)
    filtered_image = filtered_image.astype(np.uint8)
    filtered_image.show()


def gaussian_filter(image, size: int, std_dev: float):
    kernel = create_gaussian_kernel(size, std_dev)
    filtered_image = signal.convolve2d(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    filtered_image.show()
