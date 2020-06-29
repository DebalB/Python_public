# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 00:27:11 2019

@author: DEBAL
"""

# USAGE
#--video1 sample/test_0.mp4 --video2 sample/test_1.mp4 --output sample/stitched_video.mp4

# import the necessary packages
from pyimagesearch.panorama import Stitcher
from imutils.perspective import four_point_transform
import argparse
import imutils
import cv2
import os
import time
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--video1", required=True,
	help="path to the left video")
ap.add_argument("-r", "--video2", required=True,
	help="path to the right video")
ap.add_argument("-o", "--output", required=True,
	help="path to the output video")
args = vars(ap.parse_args())

# grab a reference to the video files
vs1 = cv2.VideoCapture(args["video1"])
vs2 = cv2.VideoCapture(args["video2"])

# allow the camera or video file to warm up
time.sleep(2.0)

#image = image_org.copy()
#image = imutils.resize(image_org, width=800)

outdir = os.path.split(args["video1"])[0]

# stitch the images together to create a panorama
stitcher = Stitcher()
fc1 = 0
fc2 = 0

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(args["output"], fourcc, 30, (2560, 960), True)

while True:
  # grab the current frame from left video
  result1, image1 = vs1.read()
  
  # if we are viewing a video and we did not grab a frame,
  # then we have reached the end of the video
  if result1 == False:
    break
  
  fc1+=1
  
  # grab the current frame from right video
  result2, image2 = vs2.read()
  
  # if we are viewing a video and we did not grab a frame,
  # then we have reached the end of the video
  if result2 == False:
    break
  
  fc2+=1
  
  (H1,W1,_) = image1.shape
  (H2,W2,_) = image2.shape

  # Left image coordinates
  top_left1 = (0, 0)
  top_right1 = (W1,0)
  bottom_right1 = (W1,H1)
  bottom_left1 = (0,H1)
  
  # Right image coordinates
  #top_left2 = (100,0)
  top_left2 = (0,0)
  top_right2 = (W2,0)
  bottom_right2 = (W2,H2)
  bottom_left2 = (20,H2)
  
  coords1 = [top_left1, top_right1, bottom_right1, bottom_left1]
  pts1 = np.array(coords1, dtype = "float32")
  
  coords2 = [top_left2, top_right2, bottom_right2, bottom_left2]
  pts2 = np.array(coords2, dtype = "float32")

  #cv2.circle(image, coords[0], 5, (0, 0, 255), -1)
  #cv2.circle(image, coords[1], 5, (0, 0, 255), -1)
  #cv2.circle(image, coords[2], 5, (0, 0, 255), -1)
  #cv2.circle(image, coords[3], 5, (0, 0, 255), -1)
  pts1_1 = np.float32([coords1[0], coords1[1], coords1[2], coords1[3]])
  pts2_1 = np.float32([[0, 0], [W1, 0], [W1, H1], [0, H1]])
  matrix1 = cv2.getPerspectiveTransform(pts1_1, pts2_1)
  result1 = cv2.warpPerspective(image1, matrix1, (W1, H1))

  pts1_2 = np.float32([coords2[0], coords2[1], coords2[2], coords2[3]])
  pts2_2 = np.float32([[0, 0], [W2, 0], [W2, H2], [0, H2]])
  matrix2 = cv2.getPerspectiveTransform(pts1_2, pts2_2)
  result2 = cv2.warpPerspective(image2, matrix2, (W2, H2))

  warped1 = four_point_transform(image1, pts1_1)
  warped1 = cv2.resize(warped1,(W1,H1))
  
  warped2 = four_point_transform(image2, pts1_2)
  warped2 = cv2.resize(warped2,(W2,H2))

  # show the warped images
  cv2.imshow("Video1", imutils.resize(warped1, width=800))
  cv2.imshow("Video2", imutils.resize(warped2, width=800))
  
  (result, vis) = stitcher.stitch([warped1, warped2], showMatches=True,ratio=0.75, reprojThresh=4.0)

  # show the images
  cv2.imshow("Keypoint Matches", imutils.resize(vis, width=1280))
  cv2.putText(result, "FC1={},FC2={}".format(fc1,fc2), (10,25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
  
  cv2.imshow("Result", imutils.resize(result, width=1280))
  
  if writer is not None:
    writer.write(result)

  #filename = os.path.split(args["image"])[-1]
  #cv2.imwrite(os.path.sep.join([outdir, filename.split('.')[0]+"_trans."+filename.split('.')[-1]]),warped)
  
  
  key = cv2.waitKey(2)
  if key == 27:
    break
  
  
cv2.waitKey(0)
cv2.destroyAllWindows()

if writer is not None:
  writer.release()

vs1.release()
vs2.release()

#outdir = os.path.split(args["first"])[0]
#filename = os.path.split(args["first"])[-1]
#cv2.imwrite(os.path.sep.join([outdir, "stitched.png"]),result)
