import numpy as np
import cv2

def normalize(img , new_max ,new_min):
    rows,cols,_ = img.shape
    # red
    min_b = img[..., 0].min()
    max_b = img[..., 0].max()
    # green
    min_g = img[...,1 ].min()
    max_g = img[..., 1].max()
    # blue
    min_r = img[..., 2].min()
    max_r = img[..., 2].max()

    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = int ( ( (img[i,j][0] - min_b) * ( (new_max - new_min) /  (max_b - min_b) ) ) + new_min)
            img[i,j][1] = int ( ( (img[i,j][1] - min_g) * ( (new_max - new_min) /   (max_g - min_g) ) ) + new_min)
            img[i,j][2] = int ( ( (img[i,j][2] - min_r) * ( (new_max - new_min) /    (max_r - min_r) ) ) + new_min)

