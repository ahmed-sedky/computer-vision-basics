import numpy as np
import cmath
import math

def fft(img):
   rows = len(img)
   if rows == 1:
      return img

   Feven, Fodd = fft(img[0::2]), fft(img[1::2])
   combined = [0] * rows
   for i in range(int(rows/2)):
     expTerm = cmath.exp((-2.0 * cmath.pi * 1j * i) / rows)
     combined[i] = Feven[i] + (expTerm * Fodd[i])
     combined[i + int(rows/2)] = Feven[i] - expTerm * Fodd[i]
   return combined

def pad2(img):
   rows, cols = np.shape(img)
   M, N = 2 ** int(math.ceil(math.log(rows, 2))), 2 ** int(math.ceil(math.log(cols, 2)))
   F = np.zeros((M,N), dtype = img.dtype)
   F[0:rows, 0:cols] = img
   return F 

def fft2(img):
   img = pad2(img)
   return np.transpose(fft(np.transpose(fft(img))))

def ifft2(img,rows,cols):
   img = fft2(np.conj(img))
   img = np.matrix(np.real(np.conj(img)))/(rows*cols)
   return img[0:rows, 0:cols]

def highPassFiltering(img,filterSize):
    rows, cols = img.shape[0:2]
    filterHeight,filterWidth = int(rows/2), int(cols/2)
    img[filterHeight-int(filterSize/2):filterHeight+int(filterSize/2), filterWidth-int(filterSize/2):filterWidth+int(filterSize/2)] = 0
    return img

def lowPassFiltering(img,size):
    rows, cols = img.shape[0:2]
    filterHeight,filterWidth = int(rows/2), int(cols/2)
    img[:filterHeight-int(size/2), :] = 0
    img[filterHeight-int(size/2):filterHeight+int(size/2), :filterWidth-int(size/2)] = 0
    img[filterHeight-int(size/2):filterHeight+int(size/2), filterWidth+int(size/2):] = 0
    img[filterHeight+int(size/2):, :] = 0
    return img

def fftshift(img):
   rows, cols = img.shape
   firstQuad, secondQuad = img[0: int(rows/2), 0: int(cols/2)], img[int(rows/2): rows, 0: int(cols/2)]
   thirdQuad, fourthQuad = img[0: int(rows/2), int(cols/2): cols], img[int(rows/2): rows, int(cols/2): cols]
   shiftedImg = np.zeros(img.shape,dtype = img.dtype)
   shiftedImg[int(rows/2): rows, int(cols/2): cols], shiftedImg[0: int(rows/2), 0: int(cols/2)] = firstQuad, fourthQuad
   shiftedImg[int(rows/2): rows, 0: int(cols/2)], shiftedImg[0: int(rows/2), int(cols/2): cols]= thirdQuad, secondQuad
   return shiftedImg
