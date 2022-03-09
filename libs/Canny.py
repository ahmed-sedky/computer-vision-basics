import numpy as np
from math import ceil, floor, pi
from libs.utils import convolution
from libs.filters import gaussian_filter

def checkNeighboringPixels(
    image, startingRow, endingRow, startingCol, endingCol, rowStep, colStep
):
    for row in range(startingRow, endingRow, rowStep):
        for col in range(startingCol, endingCol, colStep):
            if image[row, col] == 100:
                for neighboringR in range(-1, 2):
                    if image[row, col] == 255:
                        break
                    for neighboringC in range(-1, 2):
                        if neighboringC == 0 and neighboringR == 0:
                            continue
                        if image[row + neighboringR, col + neighboringC] == 255:
                            image[row, col] = 255
                            break
                if image[row, col] != 255:
                    image[row, col] = 0
    return image


def canny(image):

    sobelHKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    image = gaussian_filter(image, 9, 1.5)
    horizontalConv = convolution(image, sobelHKernel)
    verticalConv = convolution(image, sobelVKernel)

    # magnitude and direction of gradients
    magnitude = np.sqrt(np.square(verticalConv) + np.square(horizontalConv))
    magnitude *= 255.0 / magnitude.max()

    direction = np.arctan2(verticalConv, horizontalConv)

    direction = np.rad2deg(direction)

    direction = 180 - direction

    # non_max_supression
    supression = np.zeros(magnitude.shape)

    for row in range(1, len(magnitude) - 1):
        for col in range(1, len(magnitude[0]) - 1):
            PI = 180
            angle = direction[row, col]
            if (
                (0 <= angle < PI / 8)
                or (15 * PI / 8 <= angle <= 2 * PI)
                or (7 * PI / 8 <= angle < 9 * PI / 8)
            ):
                previous_pixel = magnitude[row, col - 1]
                next_pixel = magnitude[row, col + 1]

            elif (PI / 8 <= angle < 3 * PI / 8) or (9 * PI / 8 <= angle < 11 * PI / 8):
                previous_pixel = magnitude[row + 1, col - 1]
                next_pixel = magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= angle < 5 * PI / 8) or (11 * PI / 8 <= angle < 13 * PI / 8):
                previous_pixel = magnitude[row - 1, col]
                next_pixel = magnitude[row + 1, col]

            else:
                previous_pixel = magnitude[row - 1, col - 1]
                next_pixel = magnitude[row + 1, col + 1]

            if magnitude[row, col] >= previous_pixel and magnitude[row, col] >= next_pixel:
                supression[row, col] = magnitude[row, col]


    # thresholding with 5 and 30
    threshold = np.zeros(supression.shape)

    strong_row, strong_col = np.where(supression >= 20)
    weak_row, weak_col = np.where((supression <= 20) & (supression >= 5))

    threshold[strong_row, strong_col] = 255
    threshold[weak_row, weak_col] = 100

    # hysteresis
    thresholdRows, thresholdCols = threshold.shape

    topToBottom = checkNeighboringPixels(
        threshold.copy(), 1, thresholdRows, 1, thresholdCols, 1, 1
    )

    bottomToTop = checkNeighboringPixels(
        threshold.copy(), thresholdRows - 1, 0, thresholdCols - 1, 0, -1, -1
    )

    rightToLeft = checkNeighboringPixels(
        threshold.copy(), 1, thresholdRows, thresholdCols - 1, 0, 1, -1
    )

    leftToRight = checkNeighboringPixels(
        threshold.copy(), thresholdRows - 1, 0, 1, thresholdCols, -1, 1
    )

    final_image = topToBottom + bottomToTop + rightToLeft + leftToRight

    final_image[final_image > 255] = 255

    return final_image
