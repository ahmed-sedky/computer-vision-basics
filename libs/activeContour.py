import itertools
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

from libs import Sobel

def calculateAreaPerimeter (contour_x: np.ndarray, contour_y: np.ndarray):
    area=0.5*np.sum(contour_y[:-1]*np.diff(contour_x) - contour_x[:-1]*np.diff(contour_y))
    return abs(area)

def iterate_contour(source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                    external_energy: np.ndarray, window_coordinates: list,
                    alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:

    src = np.copy(source)
    cont_x = np.copy(contour_x)
    cont_y = np.copy(contour_y)

    contour_points = len(cont_x)

    for Point in range(contour_points):
        MinEnergy = np.inf
        TotalEnergy = 0
        NewX = None
        NewY = None
        for Window in window_coordinates:
            # Create Temporary Contours With Point Shifted To A Coordinate
            CurrentX, CurrentY = np.copy(cont_x), np.copy(cont_y)
            CurrentX[Point] = CurrentX[Point] + Window[0] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
            CurrentY[Point] = CurrentY[Point] + Window[1] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

            # Calculate Energy At The New Point
            try:
                TotalEnergy = - external_energy[CurrentY[Point], CurrentX[Point]] + calculate_internal_energy(CurrentX,
                                                                                                              CurrentY,
                                                                                                              alpha,
                                                                                                              beta)
            except:
                pass

            # Save The Point If It Has The Lowest Energy In The Window
            if TotalEnergy < MinEnergy:
                MinEnergy = TotalEnergy
                NewX = CurrentX[Point] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
                NewY = CurrentY[Point] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

        # Shift The Point In The Contour To It's New Location With The Lowest Energy
        cont_x[Point] = NewX
        cont_y[Point] = NewY

    return cont_x, cont_y


def create_square_contour(source, num_xpoints, num_ypoints):
    step = 5

    # Create x points lists
    t1_x = np.arange(0, num_xpoints, step)
    t2_x = np.repeat((num_xpoints) - step, num_xpoints // step)
    t3_x = np.flip(t1_x)
    t4_x = np.repeat(0, num_xpoints // step)

    # Create y points list
    t1_y = np.repeat(0, num_ypoints // step)
    t2_y = np.arange(0, num_ypoints, step)
    t3_y = np.repeat(num_ypoints - step, num_ypoints // step)
    t4_y = np.flip(t2_y)

    # Concatenate all the lists in one array
    contour_x = np.array([t1_x, t2_x, t3_x, t4_x]).ravel()
    contour_y = np.array([t1_y, t2_y, t3_y, t4_y]).ravel()

    # Shift the shape to a specific location in the image
    # contour_x = contour_x + (source.shape[1] // 2) - 85
    contour_x = contour_x + (source.shape[1] // 2) - 95
    contour_y = contour_y + (source.shape[0] // 2) - 40

    # Create neighborhood window
    WindowCoordinates = GenerateWindowCoordinates(5)

    return contour_x, contour_y, WindowCoordinates


def create_elipse_contour(source, num_points):
    # Create x and y lists coordinates to initialize the contour
    t = np.arange(0, num_points / 10, 0.1)

    #  Coordinates for Circles.png image
    # contour_x = (source.shape[1] // 2) + 117 * np.cos(t) - 100
    # contour_y = (source.shape[0] // 2) + 117 * np.sin(t) + 50

    # Coordinates for fish.png image
    contour_x = (source.shape[1] // 2) + 215 * np.cos(t)
    contour_y = (source.shape[0] // 2) + 115 * np.sin(t) - 10
    contour_x = contour_x.astype(int)
    contour_y = contour_y.astype(int)

    # Create neighborhood window
    WindowCoordinates = GenerateWindowCoordinates(5)

    return contour_x, contour_y, WindowCoordinates


def GenerateWindowCoordinates(Size: int):
    # Generate List of All Possible Point Values Based on Size
    Points = list(range(-Size // 2 + 1, Size // 2 + 1))
    PointsList = [Points, Points]

    # Generates All Possible Coordinates Inside The Window
    Coordinates = list(itertools.product(*PointsList))
    return Coordinates


def calculate_internal_energy(CurrentX, CurrentY, alpha: float, beta: float):
    JoinedXY = np.array((CurrentX, CurrentY))
    Points = JoinedXY.T

    # Continuous  Energy
    PrevPoints = np.roll(Points, 1, axis=0)
    NextPoints = np.roll(Points, -1, axis=0)
    Displacements = Points - PrevPoints
    PointDistances = np.sqrt(Displacements[:, 0] ** 2 + Displacements[:, 1] ** 2)
    MeanDistance = np.mean(PointDistances)
    ContinuousEnergy = np.sum((PointDistances - MeanDistance) ** 2)

    # Curvature Energy
    CurvatureSeparated = PrevPoints - 2 * Points + NextPoints
    Curvature = (CurvatureSeparated[:, 0] ** 2 + CurvatureSeparated[:, 1] ** 2)
    CurvatureEnergy = np.sum(Curvature)

    return alpha * ContinuousEnergy + beta * CurvatureEnergy

def calculate_external_energy(source, WLine, WEdge):
    src = np.copy(source)
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        gray = src

    ELine = cv2.GaussianBlur(gray,(7,7),7)	
    EEdge = Sobel.sobel(ELine)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(EEdge, cmap='gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlim(0, src.shape[1])
    # ax.set_ylim(src.shape[0], 0)
    # plt.show()
    return WLine * ELine + WEdge * EEdge[1:-1,1:-1]

