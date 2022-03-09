import matplotlib.pyplot as plt
import numpy as np
from libs.utils import (
    frequencies_of_pixel_values,
    get_pixel_values,
    image_mean,
    image_standard_deviation,
)


def histogram(image):
    if (len(image.shape)<2):
        grayscale_histogram(image)
    elif (len(image.shape)==3):
        rgb_histogram(image)


def grayscale_histogram(image):
    frequencies = frequencies_of_pixel_values(image)
    frequencies = {key: frequencies[key] for key in sorted(frequencies)}
    pixels = list(frequencies.keys())
    occurences = list(frequencies.values())

    plt.plot(pixels, occurences, color="black")
    plt.xlabel("Grayscale values")
    plt.ylabel("No. of pixels")
    plt.show()

    return frequencies


def rgb_histogram(image):
    frequencies = frequencies_of_pixel_values(image)
    pixels = [[], [], []]
    occurences = [[], [], []]
    colors = ["blue", "green", "red"]
    for k in range(3):
        frequencies[k] = {key: frequencies[k][key] for key in sorted(frequencies[k])}
        pixels[k] = list(frequencies[k].keys())
        occurences[k] = list(frequencies[k].values())
        plt.plot(
            list(frequencies[k].keys()), list(frequencies[k].values()), color=colors[k]
        )

    plt.xlabel("RGB values")
    plt.ylabel("No. of occurences")
    plt.show()
    return frequencies


def distribution_curve(image):
    if (len(image.shape)<2):
        grayscale_distribution_curve(image)
    elif (len(image.shape)==3):
        rgb_distribution_curve(image)


def grayscale_distribution_curve(image):
    pixel_values = get_pixel_values(image)
    mean = image_mean(image)
    standard_deviation = image_standard_deviation(image)
    probability_density = (np.pi * standard_deviation) * np.exp(
        -0.5 * ((np.asarray(sorted(pixel_values)) - mean) / standard_deviation) ** 2
    )

    plt.plot(sorted(pixel_values), probability_density, color="black")
    plt.xlabel("Pixel values")
    plt.ylabel("Probability Density")
    plt.show()


def rgb_distribution_curve(image):
    rgb_values = get_pixel_values(image)
    mean = image_mean(image)
    standard_deviation = image_standard_deviation(image)
    colors = ["blue", "green", "red"]
    for i in range(3):
        probability_density = (np.pi * standard_deviation[i]) * np.exp(
            -0.5
            * ((np.asarray(sorted(rgb_values[i])) - mean[i]) / standard_deviation[i])
            ** 2
        )
        plt.plot(sorted(rgb_values[i]), probability_density, color=colors[i])
        
    plt.xlabel("RGB values")
    plt.ylabel("Probability Density")
    plt.show()
