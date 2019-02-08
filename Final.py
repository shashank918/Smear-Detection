
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import imutils
import sys
import glob

def SmearDetector():
	arg = sys.argv
	if len(arg) < 2:
		print("Please specify arguments: Where images are kept...")
		print("USAGE: python Final.py <Images path>/*.jpg")
		exit(1)
	try:
		start = time.strftime("%c")
		## Program start time
		print("Program starts: " + start)
		print("")
		imageavg = np.zeros((500,500),np.float)
		#load all images from directory
		images = glob.glob(arg[1])
		#print(images)
		#for averaging the image
		for img in images:
			readimg = cv2.imread(img)
			imageresize = imutils.resize(readimg, width=500)
			imagegray = cv2.cvtColor(imageresize,cv2.COLOR_BGR2GRAY)
			imagehist = cv2.equalizeHist(imagegray)
			imageavg = imageavg + imagehist

		imageavg = imageavg/len(images)
		imageavg = np.array(np.round(imageavg), dtype=np.uint8)

		cv2.imwrite("Average.jpg", imageavg)
	except FileNotFoundError:
		print("Please Enter correct path")
	
	#Reading the averaged image
	image = cv2.imread('Average.jpg',0)
	#Adding adaptive threshold
	imagethresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,4)

	#softening the image by removing salt-pepper noise
	imagemedblr = cv2.medianBlur(imagethresh,29)
	warped = imagemedblr.astype("uint8") * 255

	#Detecting the edges in the image
	image_ed = cv2.Canny(warped, 9,50,apertureSize=5,L2gradient=True)

	#Detecting the Contours
	imagecon, contours, hierarchy = cv2.findContours(image_ed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	laplacian = cv2.Laplacian(image,cv2.CV_64F)
	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)

	plt.imshow(sobelx,cmap = 'gray')
	plt.axis('off')
	plt.savefig('Gradient.jpg', bbox_inches='tight', pad_inches = 0)

	list = []
	for i in contours:
		list.append(i)
	mask = np.zeros((500,500,1),np.float)
	
	imageread1 = cv2.imread(glob.glob(arg[1])[2000])
	imageread2 = cv2.imread(glob.glob(arg[1])[2000])

	imageresize2 = imutils.resize(imageread2,width=500)
	imageresize1 = imutils.resize(imageread1,width=500)
	
	imageread3 = cv2.imread('Average.jpg')
	if len(list) > 0:		
		k = cv2.drawContours(mask, contours, -1, (255, 255, 255), 15)
		imgecontour3 = cv2.drawContours(imageread3, contours, -1, (0, 255, 0), 3)

		cv2.imwrite('SmearOnAverageImage.jpg',imgecontour3)
		cv2.imwrite('MaskedImage.jpg',mask)
		
		img = cv2.drawContours(imageresize1, contours, -1, (0,255,0), 3)
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Resultant Image with smear'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,3),plt.imshow(imageresize2,cmap = 'gray')
		plt.title('Original Image without smear detected'), plt.xticks([]), plt.yticks([])
		plt.axis('off')
		plt.savefig('FinalResult.jpg')
		plt.show()
		print("Smear Detected. Result in FinalResult.jpg")
	else:
		print("Smear not detected")
	
if __name__ == '__main__':
    SmearDetector()