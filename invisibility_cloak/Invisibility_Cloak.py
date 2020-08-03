import cv2
import numpy as np
import time
import argparse
import imutils

parser = argparse.ArgumentParser()
# Input argument
parser.add_argument("--video", help = "Path to input video file. Skip this argument to capture frames from a camera.")

args = parser.parse_args()

print("""

Harry :  Hey !! Would you like to try my invisibility cloak ??

         Its awesome !!

        
         Prepare to get invisible .....................
    """)

# # RED color
# H_low1 = 0
# H_high1 = 10
# S_low1 = 120
# S_high1 = 255
# V_low1 = 70
# V_high1 = 255

# H_low2 = 170
# H_high2 = 180
# S_low2 = 120
# S_high2 = 255
# V_low2 = 70
# V_high2 = 255

# Pink color
# H_low1 = 0
# H_high1 = 10
# S_low1 = 120
# S_high1 = 255
# V_low1 = 70
# V_high1 = 255

# H_low2 = 170
# H_high2 = 180
# S_low2 = 120
# S_high2 = 255
# V_low2 = 70
# V_high2 = 255

# # Blue color
H_low1 = 99
H_high1 = 157
S_low1 = 83
S_high1 = 255
V_low1 = 40
V_high1 = 255

H_low2 = 99
H_high2 = 157
S_low2 = 83
S_high2 = 255
V_low2 = 40
V_high2 = 255

# Creating an VideoCapture object
# This will be used for image acquisition later in the code.
cap = cv2.VideoCapture(args.video if args.video else 0)

# We give some time for the camera to setup
time.sleep(3)
count = 0
background=0

# Capturing and storing the static background frame
for i in range(60):
	ret,background = cap.read()

#background = np.flip(background,axis=1)

while(cap.isOpened()):
	ret, img = cap.read()
	if not ret:
		break
	count+=1
	#img = np.flip(img,axis=1)
	
	# Converting the color space from BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Generating mask to detect red color
	lower_red = np.array([H_low1,S_low1,V_low1])
	upper_red = np.array([H_high1,S_high1,V_high1])
	mask1 = cv2.inRange(hsv,lower_red,upper_red)

	lower_red = np.array([H_low2,S_low2,V_low2])
	upper_red = np.array([H_high2,S_high2,V_high2])
	mask2 = cv2.inRange(hsv,lower_red,upper_red)

	mask1 = mask1+mask2

	# Refining the mask corresponding to the detected red color
	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
	mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
	mask2 = cv2.bitwise_not(mask1)

	# Generating the final output
	res1 = cv2.bitwise_and(background,background,mask=mask1)
	res2 = cv2.bitwise_and(img,img,mask=mask2)
	final_output = cv2.addWeighted(res1,1,res2,1,0)

# 	cv2.imshow('Magic !!!',final_output)
	cv2.imshow('Magic !!!',imutils.resize(final_output, width=1024))
	k = cv2.waitKey(10)
	if k == 27:
		break
