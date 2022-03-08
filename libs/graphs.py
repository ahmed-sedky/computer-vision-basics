from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from libs.utils import rgb_to_bw, image_mean, image_standard_deviation


def histogram(image):
    if(len(image.shape)<3):
      grayscale_histogram(image)
    elif len(image.shape)==3:
      rgb_histogram(image)


def grayscale_histogram(image):
    frequency = {}
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            frequency[pixel] = (
                frequency[pixel] + 1 if (pixel in frequency) else 1
            )

    frequency = {key: frequency[key] for key in sorted(frequency)}
    pixels = list(frequency.keys())
    occurences = list(frequency.values())

    plt.plot(pixels, occurences)
    plt.xlabel("Grayscale values")
    plt.ylabel("No. of pixels")
    plt.show()

def rgb_histogram(image):
    frequency = [{}, {}, {}]
    height, width = image.shape[:2]
    pixels = [[], [], []]
    occurences = [[], [], []]
    colors = ["blue", "green", "red"]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                pixel = image[i, j][k]
                frequency[k][pixel] = (
                    frequency[k][pixel] + 1 if (pixel in frequency[k]) else 1
                )
    for k in range(3):
        frequency[k] = {key: frequency[k][key] for key in sorted(frequency[k])}
        pixels[k] = list(frequency[k].keys())
        occurences[k] = list(frequency[k].values())
        plt.plot(list(frequency[k].keys()), list(frequency[k].values()), color = colors[k])
    plt.xlabel("RGB values")
    plt.ylabel("No. of occurences")
    plt.show()


def distribution_curve(image):
    mean = image_mean(image)
    standard_deviation = image_standard_deviation(image)
    probability_density = (np.pi * standard_deviation) * np.exp(
        -0.5 * ((image.shape[:2] - mean) / standard_deviation) ** 2
    )
    plt.plot(image.shape[:2], probability_density, color="red")
    plt.xlabel("Data points")
    plt.ylabel("Probability Density")
    plt.show()
