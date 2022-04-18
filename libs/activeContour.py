import cv2
import numpy as np
import math
# from edge_detection import sobel
from scipy import signal,ndimage

import utils

Wline = 0
Wterm = 0
alpha = 0.1
beta = 0.1

def imagederivative(img,sigma,typeD):
  x = np.zeros((49,49))
  for i in range(49):
      for j in range(49):
          x[i][j] = j - 24
  x = x.T
  y = x.T
  dgauss = x      
  if (typeD=='x'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=-1*(x[i][j]/(2*math.pi*(sigma**4)))*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*(sigma**2)))
    #   for i in range(5):
    #       print(dgauss[15][30+i])      
    #   im2 = cv2.imread('BasicSnake_version2f/testimage2.png')
    #   im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    #   cv2.imwrite("der.jpg",signal.convolve2d(im2,dgauss,boundary='symm'))
  elif (typeD=='y'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=-1*(y[i][j]/(2*math.pi*sigma**4))*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*sigma**2))
  elif (typeD=='xx'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*math.pi*sigma**4))(x[i][j]**2/sigma**2 - 1)*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*sigma**2))
  elif (typeD=='xy' or typeD=='yx'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*np.pi*sigma**4))(x[i][j]*y[i][j])*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*sigma**2))
  else:
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*np.pi*sigma**4))(y[i][j]**2/sigma**2 - 1)*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*sigma**2))

  return signal.convolve2d(img,dgauss,boundary='symm')      


def move(matrix,contour,fext,gamma,kappa,delta,itr):
    contour[0][contour[0]<1] = 1
    contour[0][contour[0]>len(fext[0])] = len(fext[0])
    
    contour[1][contour[1]<1] = 1
    contour[1][contour[1]>len(fext[0][1])] = len(fext[1][0])

    fext1 = [[],[]]
    fext1[0] = kappa*ndimage.map_coordinates(fext[0],contour)
    fext1[1] = kappa*ndimage.map_coordinates(fext[1],contour)
    if(itr==0):
        print(fext1[0][50])
    if(itr==50):
        print(fext1[0][50])
    ssx = np.matrix(gamma*contour[0] + fext1[0])
    ssy = np.matrix(gamma*contour[1] + fext1[1])
    if np.shape(ssx)[0]==1:
        ssx = ssx.T 
    if np.shape(ssy)[0]==1:
        ssy = ssy.T 

    contour[0] = matrix * ssx;
    contour[1] = matrix * ssy;

    contour[0][contour[0]<1] = 1
    contour[0][contour[0]>len(fext[0])] = len(fext[0])

    contour[1][contour[1]<1] = 1
    contour[1][contour[1]>len(fext[0][1])] = len(fext[1][0])

    return contour

def get_external_energy(image):
    Ix = imagederivative(image,8,'x')
    Iy = imagederivative(image,8,'y')

    return -2*np.sqrt(np.gradient(image)[0]**2 + np.gradient(image)[1]**2)


image = cv2.imread('BasicSnake_version2f/testimage2.png')

t = np.arange(0,2*np.pi ,0.05)
x = int(np.shape(image)[0]/2)+130*np.cos(t)
y = int(np.shape(image)[1]/2)+130*np.sin(t)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resultImage = np.zeros((np.shape(image)[0],np.shape(image)[1]))
img = cv2.normalize(image.astype('float'), resultImage, 0.0, 1.0, cv2.NORM_MINMAX)

external_energy = get_external_energy(img)

fx = np.gradient(external_energy)[0]
fy = np.gradient(external_energy)[1]
# cv2.imwrite("external.jpg",cv2.normalize(fy.astype('float'), resultImage, 0.0, 255, cv2.NORM_MINMAX))

fext = [[],[]]
fext[0] = -1*fx*(2*8**2)
fext[1] = -1*fy*(2*8**2)

fext = np.array(fext)

matrix = []
first_row = np.zeros(len(x))
first_row[0] = (2*alpha + 6*beta)
first_row[1] = -1*(alpha + 4*beta)
first_row[2] = beta
first_row[-1] = first_row[1]
first_row[-2] = first_row[2]

matrix.append(first_row)
for i in range(len(x) - 1):
    matrix.append(np.roll(first_row,i))

matrix = np.matrix(matrix)
matrix = np.linalg.inv( matrix + 1*np.identity(len(x)))

contour = [np.array(x),np.array(y)]

for i in range(300):    
    contour = move(matrix,contour,fext,1,4,0.1,i)
    img2 = image.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    result = []
    for j in range(len(contour[0])):
        result = cv2.circle(img2,(int(contour[0][j]),int(contour[1][j])),3,[255,0,0],-1)
    cv2.imwrite("result" + str(i) + ".jpg",result)