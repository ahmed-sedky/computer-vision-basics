import numpy as np


def global_threshold(img, T0):
    row, column = img.shape
    sum_above = sum_below = no_above = no_below = 0
    for y in range(0, row):
        for x in range(0, column):
            if img[y, x] > T0:
                sum_above = sum_above + img[y, x]
                no_above += 1
            else:
                sum_below = sum_below + img[y, x]
                no_below += 1

    mean_above = sum_above / no_above
    mean_below = sum_below / no_below

    T_new = (mean_above + mean_below) / 2
    if abs(T_new - T0) > 0.5:
        global_threshold(img, T_new)
    else:
        for y in range(0, row):
            for x in range(0, column):
                if img[y, x] > T0:
                    img[y, x] = 255
                else:
                    img[y, x] = 0

    return img


def local_threshold(img, threshold_value):
    threshold_img = np.zeros_like(img)
    row, column = img.shape
    int_img = np.zeros_like(img, dtype=np.uint32)

    for y in range(column):
        for x in range(row):
            int_img[x, y] = img[0:x, 0:y].sum()

    s = column / 32

    for y in range(column):
        for x in range(row):
            y1 = int(max(x - s, 0))
            y2 = int(min(x + s, row - 1))
            x1 = int(max(y - s, 0))
            x2 = int(min(y + s, column - 1))

            count = (y2 - y1) * (x2 - x1)
            sum_ = int_img[y2, x2] - int_img[y1, x2] - int_img[y2, x1] + int_img[y1, x1]

            if img[x, y] * count < sum_ * (100.0 - threshold_value) / 100.0:
                threshold_img[x, y] = 0
            else:
                threshold_img[x, y] = 255

    return threshold_img


def threshold(img, type_of_threshold, thres_value = 0):
    if type_of_threshold == "global":
        return global_threshold(img, thres_value)
    else:
        return local_threshold(img, thres_value)
