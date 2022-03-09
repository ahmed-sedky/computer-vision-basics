import numpy as np
import matplotlib.pyplot as plt
from libs import graphs


def Histogram(image):
    hist = np.zeros(shape=(256, 1))
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            gray_value = image[i, j]
            hist[gray_value, 0] = hist[gray_value, 0] + 1

    plt.plot(hist)
    plt.xlabel("gray_values")
    plt.ylabel("no. of pixels")
    return hist

def histogram_equaliztion_gray(img ):
    his = Histogram (img)
    shape =img.shape
    x = his.reshape(1,256)
    y = np.array([])
    y = np.append(y, x[0, 0])

    for i in range(255):
        k = x[0, i + 1] + y[i]
        y = np.append(y, k)
    y = np.round((y / (shape[0] * shape[1])) * 255)

    for i in range(shape[0]):
        for j in range(shape[1]):
            k = img[i, j]
            img[i, j] = y[k]

def histogram_equaliztion_rgb(img ):
    his = Histogram (img)
    rows,cols,channels =img.shape
    x = his.reshape(1,256)
    y = np.array([])
    y = np.append(y, x[0, 0])

    for i in range(255):
        k = x[0, i + 1] + y[i]
        y = np.append(y, k)
    y = np.round((y / (rows * cols)) * 255)

    for i in range(rows):
        for j in range(cols):
            for m in range (channels):
                k = img[i, j,m]
                img[i, j ,m] = y[k]
def equalize(img):
    if img.shape == 3:
        histogram_equaliztion_rgb(img)
    else:
        histogram_equaliztion_gray(img)