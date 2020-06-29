# -*- coding: utf-8 -*-
# USAGE
#--image sample/frame_0.png
#--image sample/frame_1.png

"""
Created on Wed Jul 31 11:56:47 2019

@author: DEBAL
"""

import cv2
import numpy as np
import argparse
import os
import imutils
from imutils.perspective import four_point_transform


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())

# load the image
image_org = cv2.imread(args["image"])
image = image_org.copy()
#image = imutils.resize(image_org, width=800)

outdir = os.path.split(args["image"])[0]
cv2.imwrite(os.path.sep.join([outdir, "img_resized.png"]),image)

(H,W,_) = image.shape

# Left image coordinates
top_left = (0, 0)
top_right = (W,0)
bottom_right = (W,H)
bottom_left = (0,H)

## Right image coordinates
##top_left = (100,0)
#top_left = (0,0)
#top_right = (W,0)
#bottom_right = (W,H)
#bottom_left = (20,H)

coords = [top_left, top_right, bottom_right, bottom_left]
pts = np.array(coords, dtype = "float32")

#cv2.circle(image, coords[0], 5, (0, 0, 255), -1)
#cv2.circle(image, coords[1], 5, (0, 0, 255), -1)
#cv2.circle(image, coords[2], 5, (0, 0, 255), -1)
#cv2.circle(image, coords[3], 5, (0, 0, 255), -1)
pts1 = np.float32([coords[0], coords[1], coords[2], coords[3]])
pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (W, H))

warped = four_point_transform(image, pts)
warped = cv2.resize(warped,(W,H))
# show the original and warped images

cv2.imshow("Original Image", imutils.resize(image, width=800))
#cv2.imshow("Perspective transformation", result)
cv2.imshow("Warped", imutils.resize(warped, width=800))

filename = os.path.split(args["image"])[-1]
cv2.imwrite(os.path.sep.join([outdir, filename.split('.')[0]+"_trans."+filename.split('.')[-1]]),warped)

key = cv2.waitKey(0)
cv2.destroyAllWindows()
