# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 
#--first sample/frame_0.png --second sample/frame_1.png
#--first sample/frame_0_trans.png --second sample/frame_1_trans.png

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
#imageA = cv2.resize(imageA, (imageA.shape[0],imageA.shape[1]//2,))
#imageB = cv2.resize(imageB, (imageB.shape[0],imageB.shape[1]//2))
#imageA = imutils.resize(imageA, width=600)
#imageB = imutils.resize(imageB, width=600)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True,ratio=0.75, reprojThresh=4.0)

# show the images
cv2.imshow("Image A", imutils.resize(imageA, width=800))
cv2.imshow("Image B", imutils.resize(imageB, width=800))
cv2.imshow("Keypoint Matches", imutils.resize(vis, width=1280))
cv2.imshow("Result", imutils.resize(result, width=1280))
cv2.waitKey(0)
cv2.destroyAllWindows()

outdir = os.path.split(args["first"])[0]
filename = os.path.split(args["first"])[-1]
cv2.imwrite(os.path.sep.join([outdir, "stitched.png"]),result)
