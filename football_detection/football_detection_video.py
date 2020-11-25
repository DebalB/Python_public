# USAGE
# --first images/objects/ --video ../videos/match1R_short.mp4

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:19:42 2019

@author: DEBAL
"""

# import the necessary packages
import numpy as np
import imutils
import cv2
import argparse
import os
import time

ratio=0.75
#ratio=1
reprojThresh=4.0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--video", required=True,
	help="path to the video file")
args = vars(ap.parse_args())

object_list = []
if os.path.isfile(args["first"]):
  print('object is file')
  object_list.append(args["first"])
else:
  print('object is directory')
  for objfile in os.listdir(args["first"]):
    object_list.append(os.path.join(args["first"], objfile))

descriptor = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create("BruteForce")

objKeyPtList = []
objFtrList = []

for obj in object_list:
  imageA = cv2.imread(obj)
#  cv2.imshow("Image A", imageA)
  
  grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)

  # detect and extract features from the image
  (kps1, featuresA) = descriptor.detectAndCompute(grayA, None)
  
  if type(featuresA) != type(None):
    kpsA = np.float32([kp.pt for kp in kps1])
    objKeyPtList.append(kpsA)
    objFtrList.append(featuresA)

print('Total objects found:',len(objFtrList))

# grab a reference to the video file
vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

while True:
  # grab the current frame
  result, imageB = vs.read()
  
  # if we are viewing a video and we did not grab a frame,
  # then we have reached the end of the video
  if result == False:
    break
  
#  cv2.imshow("Image B", imutils.resize(imageB, width=800))
  
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

  # detect and extract features from the image
  (kps2, featuresB) = descriptor.detectAndCompute(grayB, None)
  kpsB = np.float32([kp.pt for kp in kps2])

  total_matches = 0
  for featuresA in objFtrList:
    # compute the raw matches and initialize the list of actual
    # matches
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
   
    matches = []
    # loop over the raw matches
    for m in rawMatches:
      # ensure the distance is within a certain ratio of each
      # other (i.e. Lowe's ratio test)
      if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))
    
#    ptsA = None
    ptsB = None
#    if len(matches) > 4:
    if len(matches) > 1:
      total_matches = total_matches + len(matches)
      # construct the two sets of points
#      ptsA = np.float32([kpsA[i] for (_, i) in matches])
      ptsB = np.float32([kpsB[i] for (i, _) in matches])
    
    if type(ptsB) != type(None):
      for idx in range(len(ptsB)):
        cv2.circle(imageB, (ptsB[idx][0],ptsB[idx][1]), 5, (0, 0, 255), -1)
    
  text = "Matches Found: {}".format(total_matches)
  if total_matches > 0:
    font = (0,255,0)
  else:
    font = (0,0,255)
    
  cv2.putText(imageB, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, font, 2, cv2.LINE_AA)
  
  cv2.imshow("Matches", imutils.resize(imageB, width=800))
  
  key = cv2.waitKey(1)
  if key == 27:
    break

cv2.destroyAllWindows()
