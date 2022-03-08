import numpy as np
import cv2
from utilities import max_min
def normalize(img , new_max ,new_min):
    min_b,max_b,min_g,max_g,min_r,max_r =  max_min.max_min(img)
    rows,cols,_= img.shape 
    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = int ( ( (img[i,j][0] - min_b) * ( (new_max - new_min) /  (max_b - min_b) ) ) + new_min)
            img[i,j][1] = int ( ( (img[i,j][1] - min_g) * ( (new_max - new_min) /   (max_g - min_g) ) ) + new_min)
            img[i,j][2] = int ( ( (img[i,j][2] - min_r) * ( (new_max - new_min) /    (max_r - min_r) ) ) + new_min)

