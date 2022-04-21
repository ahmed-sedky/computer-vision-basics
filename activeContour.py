import cv2
import numpy as np
from libs import filters,edge_detection
import math
# from edge_detection import sobel
from scipy import signal,ndimage
from scipy.ndimage import gaussian_filter, laplace
Wline = 0
Wterm = 0
alpha = 2 #1
beta = 0.2

def  ExternalForceImage2D(I,Wline, Wedge, Wterm,Sigma):  

    Ix=imagederivative(I,Sigma,'x')
    Iy=imagederivative(I,Sigma,'y')

    Eedge = np.sqrt(Ix**2 + Iy**2); 

    Eextern= -1.0 *Wedge*Eedge 

def makeContourClockwise (contour):
    x=contour[0]
    y=contour[1]
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    if area>0:
        print ("it's counter clockwise ")

def GVFOptimizeImageForces2D(Fext, Mu, Iterations, Sigma):
    Fx= Fext[0]
    Fy= Fext[1]

    sMag = (Fx**2) + (Fy**2)

    # Set new vector-field to initial field
    u=Fx
    v=Fy
    
    # Iteratively perform the Gradient Vector Flow (GVF)
    for i in range(Iterations):
        # Calculate Laplacian
        Uxx = imagederivative(u,Sigma,'xx')
        Uyy = imagederivative(u,Sigma,'yy')
        
        Vxx = imagederivative(v,Sigma,'xx')
        Vyy = imagederivative(v,Sigma,'yy')

        # Update the vector field
        u = u + Mu*(Uxx+Uyy) - sMag*(u-Fx)
        v = v + Mu*(Vxx+Vyy) - sMag*(v-Fy)

    Fext[0] = u
    Fext[1] = v


def imagederivative(img,sigma,typeD):

  [x,y] = np.meshgrid(range(-3 * sigma, 3 * sigma), range(-3 * sigma, 3 * sigma))
  if (typeD=='x'):
    dgauss= -1*(x/(2*math.pi*(sigma**4)))*np.exp(-1*(x**2+y**2)/(2*(sigma**2)))
  elif (typeD=='y'):
    dgauss= -1*(y/(2*math.pi*(sigma**4)))*np.exp(-1*(x**2+y**2)/(2*(sigma**2)))
  elif (typeD=='xx'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*math.pi*(sigma**4))) * ((x[i][j]**2/sigma**2) - 1)*np.exp(-1*((x[i][j])**2+(y[i][j])**2)/(2*(sigma**2)))
  elif (typeD=='xy' or typeD=='yx'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*np.pi*sigma**6))*(x[i][j]*y[i][j])*np.exp(-1*((x[i][j]**2) + (y[i][j]**2))/(2*(sigma**2)))
  elif (typeD=='yy'):
      for i in range(49):
          for j in range(49):
            dgauss[i][j]=(1/(2*np.pi*sigma**4))*(y[i][j]**2/sigma**2 - 1)*np.exp(-1*(x[i][j]**2+y[i][j]**2)/(2*sigma**2))
#   return cv2.filter2D(image, -1, dgauss)       

  return signal.convolve2d(img,dgauss,boundary='symm')      


def move(matrix,contour,fext,gamma,kappa,delta,itr):
    contour[0][contour[0]<1] = 1
    contour[0][contour[0]>len(fext[0])] = len(fext[0])
    
    contour[1][contour[1]<1] = 1
    contour[1][contour[1]>len(fext[0][1])] = len(fext[1][0])
    print(contour[0])
    fext1 = [[],[]]
    fext1[0] = kappa*ndimage.map_coordinates(fext[0],contour)
    fext1[1] = kappa*ndimage.map_coordinates(fext[1],contour)
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

    return -2*np.sqrt(Ix**2 + Iy**2)


image = cv2.imread('images/hand.png')
t = np.arange(0,2*np.pi ,0.03)
x = int(np.shape(image)[0]/2)+250*np.cos(t)
y = int(np.shape(image)[1]/2)+250*np.sin(t)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resultImage = np.zeros((np.shape(image)[0],np.shape(image)[1]))
img = cv2.normalize(image.astype('float'), resultImage, 0.0, 1.0, cv2.NORM_MINMAX)

external_energy = get_external_energy(img)
# print (external_energy)
fx = imagederivative(external_energy,8,'x')# np.gradient(external_energy)[0]
fy = imagederivative(external_energy,8,'y') # np.gradient(external_energy)[1]
cv2.imwrite("external.jpg",cv2.normalize(external_energy.astype('float'), resultImage, 0.0, 255, cv2.NORM_MINMAX))

fext = [[],[]]
fext[0] = -1.0*fx*2*(8**2)
fext[1] = -1.0*fy*2*(8**2)
cv2.imwrite("fext.jpg",cv2.normalize(fext[0], resultImage, 0.0, 255, cv2.NORM_MINMAX))
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
makeContourClockwise(contour)
for i in range(300):    
    contour = move(matrix,contour,fext,1,4,0.1,i)
    img2 = image.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    result = []
    for j in range(len(contour[0])):
        result = cv2.circle(img2,(int(contour[0][j]),int(contour[1][j])),3,[255,0,0],-1)
    cv2.imwrite("result" + str(i) + ".jpg",result)