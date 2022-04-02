from math import pi
import numpy as np
from scipy import signal
from libs.edge_detection import sobel
from libs.utils import median, convolution, rgb_to_bw


def create_gaussian_kernel(kernel_size: int, std_dev: float):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * pi * std_dev ** 2))
        * np.exp(
            (-1 * ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
            / (2 * std_dev ** 2)
        ),
        (kernel_size, kernel_size),
    )
    return kernel / np.sum(kernel)

def average_filter(image, kernel_size: int):
    kernel = np.ones([kernel_size, kernel_size], dtype=int)
    kernel = kernel / (pow(kernel_size, 2))

    filtered_image = convolution(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image


def median_filter(image):
    rows, cols = image.shape
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
    return filtered_image


def gaussian_filter(image, size: int, std_dev: float):
    kernel = create_gaussian_kernel(size, std_dev)
    filtered_image = convolution(image, kernel)
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image

def hybrid (image1 ,image2):
    gaussian_blurred_image = gaussian_filter(image1,21,3)
    sobel_img , _ = sobel(image2)
    hybrid_img = sobel_img + gaussian_blurred_image
    return hybrid_img

def grayscale(image):
    copied_image = image.copy()
    width, height = copied_image.shape[:2]
    grayscale_image = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            grayscale_image[i, j] = rgb_to_bw(copied_image[i, j])
    
    return grayscale_image
