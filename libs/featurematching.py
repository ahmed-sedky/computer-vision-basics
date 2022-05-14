import SIFT
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

  
def matching(descriptor1 , descriptor2 , match_calculator):
    
    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    for kp1 in range(keypoints1):

        distance = -np.inf
        y_index = -1
        for kp2 in range(keypoints2):

         
            value = match_calculator(descriptor1[kp1], descriptor2[kp2])

            if value > distance:
              distance = value
              y_index = kp2
        
        match = cv2.DMatch()
        match.queryIdx = kp1
        match.trainIdx = y_index
        match.distance = distance
        matches.append(match)
    matches= sorted(matches, key=lambda x: x.distance, reverse=True)
    return matches
 

def calculate_ncc(descriptor1 , descriptor2):


    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))

    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    correlation = float(np.mean(correlation_vector))

    return correlation


def calculate_ssd(descriptor1 , descriptor2):

    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd = - (np.sqrt(ssd))
    return ssd






def main():
    img1 = cv2.imread("C:/Users/Mo/Desktop/CV/lenna.png", 0)
    img2 = cv2.imread("C:/Users/Mo/Desktop/CV/lenna2.png" ,0)


    # generate keypoints and descriptor using sift

    keypoints_1, descriptor1 = SIFT.SIFT(img1)
    keypoints_2, descriptor2 = SIFT.SIFT(img2)



    start_ncc = time.time()
    matches_ncc = matching(descriptor1, descriptor2, calculate_ncc)
    matched_image_ncc = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                    matches_ncc[:30], img2, flags=2)
    end_ncc = time.time()

    start_ssd = time.time()
    matches_ssd = matching(descriptor1, descriptor2, calculate_ssd)
    matched_image_ssd = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                    matches_ssd[:30], img2, flags=2)
    end_ssd = time.time()
    ssd_time = end_ssd - start_ssd
    ncc_time =  end_ncc - start_ncc

    plt.imshow( matched_image_ncc )
    plt.show()
    plt.imshow( matched_image_ssd )
    plt.show()



if __name__ == "__main__":
    main()  