import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imshow
# from Histogram import global_threshold, normalize_histogram
# from EdgeDetection import DoubleThreshold

def DoubleThreshold(Image, LowThreshold, HighThreshold, Weak, isRatio=True):
    """
       Apply Double Thresholding To Image
       :param Image: Image to Threshold
       :param LowThreshold: Low Threshold Ratio/Intensity, Used to Get Lowest Allowed Value
       :param HighThreshold: High Threshold Ratio/Intensity, Used to Get Minimum Value To Be Boosted
       :param Weak: Pixel Value For Pixels Between The Two Thresholds
       :param isRatio: Deal With Given Values as Ratios or Intensities
       :return: Threshold Image
       """

    # Get Threshold Values
    if isRatio:
        High = Image.max() * HighThreshold
        Low = Image.max() * LowThreshold
    else:
        High = HighThreshold
        Low = LowThreshold

    # Create Empty Array
    ThresholdedImage = np.zeros(Image.shape)

    Strong = 255
    # Find Position of Strong & Weak Pixels
    StrongRow, StrongCol = np.where(Image >= High)
    WeakRow, WeakCol = np.where((Image <= High) & (Image >= Low))

    # Apply Thresholding
    ThresholdedImage[StrongRow, StrongCol] = Strong
    ThresholdedImage[WeakRow, WeakCol] = Weak

    return ThresholdedImage




def GetInitialThreshold(source: np.ndarray):
    """
    Gets The Initial Threshold Used in The Optimal Threshold Method
    :param source: NumPy Array of The Source Grayscale Image
    :return Threshold: Initial Threshold Value
    """
    # Maximum X & Y Values For The Image
    MaxX = source.shape[1] - 1
    MaxY = source.shape[0] - 1
    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    BackMean = (int(source[0, 0]) + int(source[0, MaxX]) + int(source[MaxY, 0]) + int(source[MaxY, MaxX])) / 4
    Sum = 0
    Length = 0
    # Loop To Calculate Mean Value of Foreground Intensity
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += source[j, i]
                Length += 1
    ForeMean = Sum / Length
    # Get The Threshold, The Average of The Mean Background & Foreground Intensities
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def GetOptimalThreshold(source: np.ndarray, Threshold):
    """
    Calculates Optimal Threshold Based on Given Initial Threshold
    :param source: NumPy Array of The Source Grayscale Image
    :param Threshold: Initial Threshold
    :return OptimalThreshold: Optimal Threshold Based on Given Initial Threshold
    """
    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    Back = source[np.where(source < Threshold)]
    # Get Foreground Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    Fore = source[np.where(source > Threshold)]
    # Mean of Background & Foreground Intensities
    BackMean = np.mean(Back)
    ForeMean = np.mean(Fore)
    # Calculate Optimal Threshold
    OptimalThreshold = (BackMean + ForeMean) / 2
    return OptimalThreshold



def optimal(source: np.ndarray):
    """
    Applies Thresholding To The Given Grayscale Image Using The Optimal Thresholding Method
    :param source: NumPy Array of The Source Grayscale Image
    :return: Thresholded Image
    """

    src = np.copy(source)


    # Calculate Initial Thresholds Used in Iteration
    print(f"src in optimal: {src}")
    print(f"src shape: {src.shape}")
    OldThreshold = GetInitialThreshold(src)
    NewThreshold = GetOptimalThreshold(src, OldThreshold)
    iteration = 0
    # Iterate Till The Threshold Value is Constant Across Two Iterations
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = GetOptimalThreshold(src, OldThreshold)
        iteration += 1
    # src[src >= 25] = 0
    # Return Thresholded Image Using Global Thresholding
    return global_threshold(src, NewThreshold)


def otsu(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Otsu's Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(source)
    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalThreshold = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    for t in range(1, 255):
        # Background Intensities Array
        Back = np.arange(0, t)
        # Object/Foreground Intensities Array
        Fore = np.arange(t, 256)
        # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
        CDF2 = np.sum(PDF[t + 1:256])
        BackMean = sum(Back * PDF[0:t]) / CDF[t]
        ForeMean = sum(Fore * PDF[t:256]) / CDF2
        # Calculate Cross-Class Variance
        Variance = CDF[t] * CDF2 * (ForeMean - BackMean) ** 2
        # Filter Out Max Variance & It's Threshold
        if Variance > MaxVariance:
            MaxVariance = Variance
            OptimalThreshold = t
    return global_threshold(src, OptimalThreshold)


def spectral(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(source)

    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            try:
                # Background Intensities Array
                Back = np.arange(0, LowT)
                # Low Intensities Array
                Low = np.arange(LowT, HighT)
                # High Intensities Array
                High = np.arange(HighT, 256)
                # Get Low Intensities CDF
                CDFL = np.sum(PDF[LowT:HighT])
                # Get Low Intensities CDF
                CDFH = np.sum(PDF[HighT:256])
                # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                # Calculate Cross-Class Variance
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                # Filter Out Max Variance & It's Threshold
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            except RuntimeWarning:
                pass
    return DoubleThreshold(src, OptimalLow, OptimalHigh, 128, False)



def global_threshold(source: np.ndarray, threshold: int):
    src = np.copy(source)
    row, column = src.shape
    for x in range(column):
        for y in range(row):
         if src[x,y] > threshold:
             src[x,y] = 1
         else:
             src[x,y] = 0

    return src


def LocalThresholding(source: np.ndarray, Regions, ThresholdingFunction):
    """
       Applies Local Thresholding To The Given Grayscale Image Using The Given Thresholding Callback Function
       :param source: NumPy Array of The Source Grayscale Image
       :param Regions: Number of Regions To Divide The Image To
       :param ThresholdingFunction: Function That Does The Thresholding
       :return: Thresholded Image
       """
    src = np.copy(source)
    YMax, XMax = src.shape
    Result = np.zeros((YMax, XMax))
    YStep = YMax // Regions
    XStep = XMax // Regions
    XRange = []
    YRange = []
    for i in range(0, Regions+1):
        XRange.append(XStep * i)

    for i in range(0, Regions+1):
        YRange.append(YStep * i)

    
    for x in range(0, Regions):
        for y in range(0, Regions):
            Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = ThresholdingFunction(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
    return Result


def IsolatedTests():
    
    img = cv2.imread("C:/Users/Mo/Desktop/CV/lenna.png" , 0)
    # thresholded_image = optimal(img)
    # thresholded_image = LocalThresholding(img, 4 , spectral)
    # thresholded_image = LocalThresholding(img, 4 , otsu)
    # thresholded_image = LocalThresholding(img, 4 , optimal)
    thresholded_image = otsu(img)
    # thresholded_image = spectral(img)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Source Image')
    ax[0].set_axis_off()
    ax[1].imshow(thresholded_image, cmap='gray')
    ax[1].title.set_text(' Thresholded image ')
    ax[1].set_axis_off()
    plt.show()


if __name__ == "__main__":
    IsolatedTests()



