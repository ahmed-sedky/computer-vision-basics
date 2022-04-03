import numpy as np
import cv2
from libs import filters,edge_detection

def houghLine(image , rhoResolution = 1 ,thetaResolution = 1):
    grayscale_image = filters.grayscale(image)
    edgedImg = edge_detection.canny(grayscale_image,100,150)
    rows = edgedImg.shape[0]
    cols = edgedImg.shape[1]

    max_dis = int(np.ceil(np.sqrt(np.square(rows) + np.square(cols)))) # cause max distance of line is the diagonal

    thetas = np.deg2rad(np.arange(start= -90.0 , stop = 90.0 ,step= thetaResolution))
    rhos = np.arange(-max_dis, max_dis , rhoResolution)
    
    accumulator =np.zeros((len(rhos) , len(thetas) ))

    for y in range (rows):
        for x in range (cols):
            if (edgedImg[y,x] > 0):
                for theta in range (len(thetas)):
                    rho = x * np.cos(thetas[theta]) + y * np.sin(thetas[theta])
                    accumulator[int(rho) + max_dis , theta] += 1
    return accumulator, thetas, rhos

def houghCircle(img, threshold, region, radius = [3,30]):
    grayscale_image = filters.grayscale(img)
    edgedImg = edge_detection.canny(grayscale_image,100,150)
    (rows, cols) = edgedImg.shape
    [R_max, R_min] = radius

    R = R_max - R_min
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X coordinate and Y coordinate resectively.
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, rows + 2 * R_max, cols + 2 * R_max))
    B = np.zeros((R_max, rows + 2 * R_max, cols + 2 * R_max))

    # Precomputing all angles to increase the speed of the algorithm
    theta = np.deg2rad(np.arange(start= 0.0 , stop = 360.0))
    edges = np.argwhere(edgedImg[:, :])  # Extracting all edge coordinates
    for val in range(R):
        r = R_min + val
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  # Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]

def hough_peaks(HoughSpace, num_peaks, nhood_size=3):
    # loop through number of peaks to identify
    indicies = []
    HoughSpaceCopy = np.copy(HoughSpace)
    for i in range(num_peaks):
        idx = np.argmax(HoughSpaceCopy) # find argmax in flattened array
        HoughSpaceIdx = np.unravel_index(idx, HoughSpaceCopy.shape) # remap to shape of H
        indicies.append(HoughSpaceIdx)

        # surpess indicies in neighborhood
        idxRhos, idxThetas = HoughSpaceIdx # first separate x, y indexes from argmax(H)
        # if idxThetas is too close to the edges choose appropriate values
        if (idxThetas - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idxThetas - (nhood_size/2)
        if ((idxThetas + (nhood_size/2) + 1) > HoughSpace.shape[1]): max_x = HoughSpace.shape[1]
        else: max_x = idxThetas + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idxRhos - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idxRhos - (nhood_size/2)
        if ((idxRhos + (nhood_size/2) + 1) > HoughSpace.shape[0]): max_y = HoughSpace.shape[0]
        else: max_y = idxRhos + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                HoughSpaceCopy[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    HoughSpace[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    HoughSpace[y, x] = 255

    return indicies, HoughSpace

def hough_lines_draw(img, indices, rhos, thetas):
    for i in range(len(indices)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        x0 = np.cos(theta) * rho
        y0 = np.sin(theta) * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-np.sin(theta)))
        y1 = int(y0 + 1000 * (np.cos(theta)))
        x2 = int(x0 - 1000 * (-np.sin(theta)))
        y2 = int(y0 - 1000 * (np.cos(theta)))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

def hough_circle_draw(circle, img):
    circleCoordinates = np.argwhere(circle)  # Extracting the circle information
    for r, x, y in circleCoordinates:
        cv2.circle(img, (y, x), r, color=(255, 0, 0), thickness=2)
        cv2.rectangle(img,(y,x),(y,x),(255,0,0),2)
    return img