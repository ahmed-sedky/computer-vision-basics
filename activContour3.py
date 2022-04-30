# This script implements Active contours for 2D image
# import Needed packages
import cv2
import math
import numpy
import scipy
import argparse
from scipy import ndimage
import matplotlib.pyplot as plt
# Function that loads the target 10 images
def loadimages():
	# 1st five are low degree tumor 
	# 2nd  five are hight degree tumor
	# initialize empty stacks
	colorimgStack , greyimgStack ,namesStack = [] ,[],[]
	# create reading path
	#mainpath = ""
	#targetimgsNames = ["ball.jpg","pen.jpg"]
	#targetimgsNames = ["s1.jpg"]
	targetimgsNames = ["s2.jpg"]
	for i in targetimgsNames:
		tempimg = cv2.imread(i)
		greytemp = cv2.cvtColor(tempimg,cv2.COLOR_BGR2GRAY)
		# append read images
		colorimgStack.append(tempimg)
		greyimgStack.append(greytemp)
		name =i.split('.')[0]
		namesStack.append(name)
	#return loaded imgs
	return colorimgStack , greyimgStack , namesStack
# Class for contour initiation per image   
class Contour():
    def __init__(self , targetimage):
        self.img = targetimage
        self.point = []

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point

    def __onclick__(self,click):
        currntClick = (click.xdata,click.ydata)
        centerx = int(currntClick[0])
        centery = int(currntClick[1])
        self.point.append((centerx,centery))
        #print self.point
        return self.point
# function that created objs of countour per image
def objCreation(imgList):
	objList = []
	# have ERROR dont know why !!
	for i in imgList:
		objList.append(Contour(i))
		#print ("obj list lenght : ", len(objList))
	return objList
# Function Draw contours 
def drawcontour(im , contourPoints, color,figuername ):
	# input :
	# -------
	# img : target image need to draw ower it the points
	# points : list of positions where contour is
	# output :
	# -------
	# Displaied image with contour
	# Helper link
	# https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/
	p1 = (0,0)
	p2 = (0,0)
    	for i ,p in enumerate(contourPoints):
        	p1 = p
       		if (i >= len(contourPoints)-1):
        		p2 = contourPoints[-1]
        	else:
        		p2 = contourPoints[i+1]
        	cv2.circle(im, p1, 2 , color, thickness=2)
        	cv2.line(im, p1, p2, color, thickness=2)
    	cv2.imshow(figuername , im)
    	cv2.waitKey(0)
    	cv2.destroyAllWindows()
    	return im
# function that saves the image 
def saveimage(img , name ):
	cv2.imwrite(name,img)
# function that initialized the contour manually 
def initContour(imgList,imName):
	# initialize empty list of lists
	# each cell have list of 2 cells [x-coords ,y-coords]
	icontours = []
	TargetObjs =[]
	TargetObjs = objCreation(imgList)
	for i,currObj in enumerate(TargetObjs):
		# initialive contour
		currContour = currObj.getCoord()
		icontours.append(currContour)
		# draw contour initialized
		imgwithContour = drawcontour(imgList[i],currContour,(0,255,0),"Initial Contour image-{0}".format(imName))
		# save image with contour
		saveimage(imgwithContour ,"{0}-init-contour.png".format(imName))
		# it works with 1st image only !! dont know why !!!
		break
	# return initial contour
	return icontours
# function that saves contour points in text file
def saveContourPoints(points,fileName):
	# open file
	f = open('{0}-init.txt'.format(fileName), 'w')
	# loop on points
	for p in points:
		# get current coords
		x = p[0]
		y = p[1] 
		# write in file the current point in Newline
		f.write('{0} {1} \n'.format(x,y))
	# close file
	f.close()
# function that loads the contour points from text 
def loadinitContour(fileName):
	conPoints = []
	# openfile
	f = open(fileName, 'r')
	# read line by line 
	content = [x.strip('\n') for x in f.readlines()]
	# split x, y
	for i ,line in enumerate(content):
		x,y = line.split()      
		#print ("x{0}:".format(i) , x)
		#print ("y{0}:".format(i) , y)
		conPoints.append((int(x),int(y)))
	# append points
	# close file
	f.close()
	#print ("lenght of contour :" , len(conPoints))
	return conPoints
# Function that calculates the internal energy
def getinternalEnergy(c , a , b):
	# Inputs : 
	#---------
	# c : contour points
	# a : alpha ( Elasticity coeffecient ) [should be small to be more elastic]
	# b : beta ( Curveture coeffcient )
	# ===============
	# output:
	#--------
	# total energy : list of Total Internal energy on each point
	#======================
	#======================
	totalEnergy = 0
	# v_last : contour point before current point 
	# v_curr : contour current point
	# v_next : contour point after current point
	# Loop on contour points
	for i in range(len(c)):
		# get current inportant values
		if(i==0):
			v_last = c[-1] # make previouse one is the last one
			v_curr = c[i]
			v_next = c[i+1]
		elif (i == (len(c)-1)):
			v_last = c[i-1]
			v_curr = c[i]
			v_next = c[0] # make next one equal 1st one
		else:
			v_last = c[i-1]
			v_curr = c[i]
			v_next = c[i+1]
		#print ("v_last :", v_last)
		#print ("v_curr :", v_curr)
		#print ("v_next :", v_next)
		#---------------------------
		# get 1st drivative
		firstDrivX = v_next[0] - v_curr[0]
		firstDrivY = v_next[1] - v_curr[1]
		firstDriv = (firstDrivX , firstDrivY)
		#print ("First Drivative :", firstDriv)
		firstDrivPowerX = math.pow(firstDriv[0] ,2)
		firstDrivPowerY = math.pow(firstDriv[1] ,2)
		firstDrivPower = (firstDrivPowerX , firstDrivPowerY)
		#print ("First Drivative power 2 :", firstDrivPower)
		# ----------------------------------
		# Calculate elasticity MAGNITUTE
		# m = squ(x^2 +y^2)
		elasticityMagnitute = math.sqrt(firstDrivPowerX + firstDrivPowerY)
		#print ("Elasticity  Magnitute :", elasticityMagnitute)
		# ----------------------------------
		# Calculate elasticity / stiffness
		#elasticity = a * elasticityMagnitute
		elasticity = a * elasticityMagnitute
		#print ("Elasticity :", elasticity)
		# ---------------------------------
		# ========================================
		####	CLACLULATE 2nd Drivative 	####
		#-----------------------------------
		secDrivX = v_next[0] - (2 * v_curr[0]) + v_last[0]
		secDrivY = v_next[1] - (2 * v_curr[1]) + v_last[1]
		secDriv = (secDrivX , secDrivY)
		#print("Secound Drivative :" ,secDriv)
		secDrivPowerX = math.pow(secDriv[0] ,2)
		secDrivPowerY = math.pow(secDriv[1] ,2)
		secDrivPower = (secDrivPowerX , secDrivPowerY)
		#print ("Secound Drivative power 2 :", secDrivPower)
		# ----------------------------------
		# Calculate curvature MAGNITUTE
		# m = squ(x^2 +y^2)
		CurvatureMagnitute = math.sqrt(secDrivPowerX + secDrivPowerY)
		#print ("Curvature  Magnitute :", CurvatureMagnitute)
		# 
		# ----------------------------------
		# Calculate curvature
		#curvature = bete * CurvatureMagnitute
		curvature = b* CurvatureMagnitute
		#print ("Curvature :", curvature)
		# ---------------------------------
		# ========================================
		####	CLACLULATE Total Energy 	####
		#-----------------------------------
		#eTotal = elasticity + curvature
		eTotal = elasticity + curvature
		#print ("Total Energy of point {0}:".format(i) , eTotal)
		totalEnergy +=  eTotal # Summition energies
	# return total every list calculated
	return totalEnergy
# function that calculated the image energy (external energy)
def getExternalEnergy(img,imgName):
	# Input :
	#--------
	# img : is grey scale image that need to calculate its energy
	# ==============
	# output :
	# --------
	# matrix == image size : represents energy value at each position pixel
	#================================================
	# Lets smooth the image using gaussian filter
	gausImg = scipy.ndimage.filters.gaussian_filter(img, sigma= 3)
	# Helper link:
	# http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
	# lets calculate gradient in X direction 
	# using soble filter in x direction
	im = gausImg.astype('int32')
	dx = ndimage.sobel(im, 0)  # horizontal derivative
	dy = ndimage.sobel(im, 1)  # vertical derivative
	mag = numpy.hypot(dx, dy)  # magnitude
	mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
	#scipy.misc.imsave('sobel-0.jpg', mag)
	saveimage(mag,'{0}-sobel.png'.format(imgName))
	return -mag
# Functon that Calculate total energy
def getTotaleEnergy(internal , external):
	return (external + internal) 
# Functon that update Contour points using energies
def updateContour(colimg ,imName, initcontour , exEnergy ,a,b):
	I = drawcontour(colimg , initcontour,(0,255,0),"{0}-init contour".format(imName))
	for iterate in range(0,2):
		# view imput image
		newContour = []	# empty list carries the points of the new contour
		updatedPoints = 0 # counter for points that changed from iteration to another
		for i , p in enumerate(initcontour):	# loop on contour points 
			currPoint = p
			#print ("currPoint :",currPoint)
			currWindowindex = []		# empty window with pixel positions
			currWindowEnergy = []		# empty window with pixel Energies (external exnergy)
			totalwindowEnergy =[]		# empty window with total energy 
			for w in range(-4,5):	# loop to fill the current window
				xtemp = currPoint[0]+w # index of current contour point + window index
				ytemp =	currPoint[1]+w
				currWindowindex.append((xtemp,ytemp)) # append position of 1st window on image
				currWindowEnergy.append(exEnergy[xtemp,ytemp]) # append value of the curr pixel energy
				#print ("currWindowindex [-1]:",currWindowindex[-1])
			#break
			for pixel in range(0,9): # loop to calculate energy at each pixel
				# calculate internal energy of contour accourding to the new point
				currPointContour = (currWindowindex[pixel][0] ,currWindowindex[pixel][1]) # get new contour point position
				currContour = initcontour # copy main contour in new one
				currContour[i] = currPointContour # update contour copy with the moved point 
				currIntEnergy = getinternalEnergy(currContour , a ,b)
				# currPixelEnergy = External energy + Internal energy
				currPixelEnergy = getTotaleEnergy( currIntEnergy , currWindowEnergy[pixel])
				#print ("exEnergy :", currWindowEnergy[pixel])
				#print("inEnergy :",currIntEnergy) 
				#print("currPixelEnergy :",currPixelEnergy) 
				totalwindowEnergy.append(currPixelEnergy)
				#print("totalwindowEnergy :",totalwindowEnergy[-1])	
			#print("Lenght totalwindowEnergy :",len(totalwindowEnergy))
			#----------------------------------------------------
			# Now lets check where is minmum energy in the window to move the point
			minEnrgyPixel = min(currWindowEnergy)
			#print("min Pixel Energy :", minEnrgyPixel)
			indexMinEnrg = currWindowEnergy.index(minEnrgyPixel)
			#print("Index of min Pixel Energy :",indexMinEnrg)
			#-------------------------------------------------------
			# compare and update contour point location
			#print("window x of min energy :" , currWindow[indexMinEnrg][0])
			#print("window y of min energy :" , currWindow[indexMinEnrg][1])
			if (indexMinEnrg!= 4): # if the min energy is not at the middle of the window
				newPointContour = (currWindowindex[indexMinEnrg][0],currWindowindex[indexMinEnrg][1])
				newContour.append(newPointContour) # move point to new location
				updatedPoints = updatedPoints +  1 	# increment counter
			else:
				newContour.append(currPoint) # keep point as it is
		print("Total number of updated points :" , updatedPoints)
		iterate +=1
		#Check changes in contour points
		#for u in range(len(initcontour)):
		#	print (" old point at {0}:".format(u) , initcontour[u])
		#	print (" new point :" , newContour[u])
	newimg = drawcontour(colimg , newContour,(255,0,0),"{0}-updated contour".format(imName))
	saveimage(newimg , "{0}-updated-Contour.png".format(imName))
	return newContour , updatedPoints
# main function for active contours
def main():
	# Load target images
	colorimgStack , greyimgStack , nameImg = loadimages()
	print ("Images loaded : DONE")
	# initialize contour
	#baseContours = initContour(colorimgStack ,nameImg)
	# Save contours in text file
	#saveContourPoints(baseContours[0],nameImg[0])
	# Load initial contours
	sample1Contour = loadinitContour('{0}-init.txt'.format(nameImg[0]))
	print("Initial contour points loaded from file : DONE")
	####	CALCULATE ENERGYIES	 #### 
	alpha = 0.0005 # Elasticity coeffecient
	beta = 15 # Curveture coeffcient
	# Calculate internal energy 
	sample1_ExternalEnergy = getExternalEnergy(greyimgStack[0],nameImg[0])
	print ("Calculate External Energy of contour : DONE")
	# ---------------------------------
	####	UPDATE	CONTOUR	POINTS 	####
	sample1_updatedContour , sample1_updatedNumb = updateContour(colorimgStack[0],nameImg[0],sample1Contour,sample1_ExternalEnergy ,alpha,beta)  
	print ("Calculate Total Energy of contour : DONE")
	
main() 