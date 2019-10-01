# USAGE
# --first images/objects/ball1.png --second images/scenes/vlcsnap-2019-07-27-15h12m55s008.png

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

ratio=0.75
#ratio=1
reprojThresh=4.0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

object_list = []
if os.path.isfile(args["first"]):
  print('object is file')
  object_list.append(args["first"])
else:
  print('object is directory')
  for objfile in os.listdir(args["first"]):
    object_list.append(os.path.join(args["first"], objfile))

scene_list = []
if os.path.isfile(args["second"]):
  print('scene is file')
  scene_list.append(args["second"])
else:
  print('object is directory')
  for scenefile in os.listdir(args["second"]):
    scene_list.append(os.path.join(args["second"], scenefile))

descriptor = cv2.xfeatures2d.SIFT_create()

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

for scn in scene_list:
  imageB = cv2.imread(scn)
  cv2.imshow("Image B", imutils.resize(imageB, width=800))
  
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

  # detect and extract features from the image
  (kps2, featuresB) = descriptor.detectAndCompute(grayB, None)
  kpsB = np.float32([kp.pt for kp in kps2])

  matches = []
  for featuresA in objFtrList:
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
   
    # loop over the raw matches
    for m in rawMatches:
      # ensure the distance is within a certain ratio of each
      # other (i.e. Lowe's ratio test)
      if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))
    
    # computing a homography requires at least 4 matches
#    ptsA = None
    ptsB = None
#    if len(matches) > 4:
    if len(matches) > 0:
      # construct the two sets of points
#      ptsA = np.float32([kpsA[i] for (_, i) in matches])
      ptsB = np.float32([kpsB[i] for (i, _) in matches])
    
      # compute the homography between the two sets of points
    #  (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    
    if type(ptsB) != type(None):
      for idx in range(len(ptsB)):
        cv2.circle(imageB, (ptsB[idx][0],ptsB[idx][1]), 5, (0, 0, 255), -1)
    
  text = "Matches Found: {}".format(len(matches))
  cv2.putText(imageB, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
  
  cv2.imshow("Matches", imutils.resize(imageB, width=800))
  key = cv2.waitKey()
  cv2.destroyAllWindows()

key = cv2.waitKey()
cv2.destroyAllWindows()
