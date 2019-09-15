# USAGE
# python pool_table_detection.py --image images/image1.png

# import the necessary packages
import argparse
import cv2
import numpy as np
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the input image and grab its dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
image = imutils.resize(image, width=800)
cv2.imshow("Image",image)

def nothing(x):
	pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
while True:
	
	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	u_v = cv2.getTrackbarPos("U - V", "Trackbars")
	maskLower = np.array([l_h, l_s, l_v])
	maskUpper = np.array([u_h, u_s, u_v])
	
	
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, maskLower, maskUpper)
	cv2.imshow("mask", mask)
	key = cv2.waitKey(1000)
	if key == 27:
		break

cv2.waitKey(0)
cv2.destroyAllWindows()
