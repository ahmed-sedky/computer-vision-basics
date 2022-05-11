import numpy as np
from math import ceil
def RGB2LUV(image):
    copied_image = image.copy()
    width, height = copied_image.shape[:2]
    for i in range(width):
        for j in range(height):
            image[i, j][0] = round(0.412453 * copied_image[i, j][0] + 0.357580 * copied_image[i, j][1] + 0.180423 * copied_image[i, j][2])
            image[i, j][1] = round(0.212671 * copied_image[i, j][0] + 0.715160 * copied_image[i, j][1] + 0.072169 * copied_image[i, j][2])
            image[i, j][2] = round(0.019334 * copied_image[i, j][0] + 0.119193 * copied_image[i, j][1] + 0.950227 * copied_image[i, j][2])
    return image