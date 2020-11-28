# -*- coding: utf-8 -*-
# Usage:
# python test.py --image images/DSC04886.JPG --pattern images/4VY1YHI4.png
# python test.py --image images/temp4.png --pattern images/4VY1YHI4.png

"""
Created on Mon Nov 11 16:41:41 2019

@author: DEBAL
"""
# In[]
import cv2
import argparse
import imutils
from imutils import paths
import numpy as np
from Matcher import Matcher
import os
import pandas as pd
from rootsift import RootSIFT

useRootSIFT = True
useMatcher = False
useFlann = False
blur_images = False
#Lowe's ratio
LRatio = 0.65
MIN_MATCH_PTS = 10
BLER_KERNEL_SIZE = (27,27)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to T-shirt image")
ap.add_argument("-p", "--pattern", required = True, help = "Path to actual print image")

def convert_rgba2rgb(image_pattern, black_bg=True):
#  cv2.imshow("image_pattern", cv2.resize(image_pattern, (800,600)))
  if pattern.shape[2] < 4:
    print('Image does not have 4 channels')
    return image_pattern
  
  src_r, src_g, src_b, src_a = cv2.split(image_pattern)
  src_r = src_r.astype(float) / 255.
  src_g = src_g.astype(float) / 255.
  src_b = src_b.astype(float) / 255.
  src_a = src_a.astype(float) / 255.
  
  if black_bg:
    bg_r = np.zeros(src_r.shape, dtype=np.uint8)
    bg_g = np.zeros(src_g.shape, dtype=np.uint8)
    bg_b = np.zeros(src_b.shape, dtype=np.uint8)
  else:
    bg_r = np.ones(src_r.shape, dtype=np.uint8)
    bg_g = np.ones(src_g.shape, dtype=np.uint8)
    bg_b = np.ones(src_b.shape, dtype=np.uint8)
    
  tgt_r = ((1 - src_a) * bg_r) + (src_a * src_r)
  tgt_g = ((1 - src_a) * bg_g) + (src_a * src_g)
  tgt_b = ((1 - src_a) * bg_b) + (src_a * src_b)
  tgt_img = cv2.merge((tgt_r, tgt_g, tgt_b))
  tgt_img = (tgt_img * 255.).astype(np.uint8)
#  cv2.imshow("tgt_img", cv2.resize(tgt_img, (800,600)))
#  cv2.imwrite("images/TrocknerDruckdaten/{}_blackbg.png".format(pattname), tgt_img)
  
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()
  return tgt_img

def matchImages(original, image_to_compare, useFlann=False):
  
  if useRootSIFT:
    # extract RootSIFT descriptors
    #print('Using RootSIFT')
    kps = None
    rs = RootSIFT()
    kp_1, desc_1 = rs.compute(original, kps)
    kp_2, desc_2 = rs.compute(image_to_compare, kps)
  else:
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
  
  if useFlann:
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks = 50)
    print('Using Flann Matcher')
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
  else:
    print('Using Brute Force Matcher')
    matcher = cv2.DescriptorMatcher_create("BruteForce")
  
  matches = matcher.knnMatch(desc_1, desc_2, k=2)
  
  good_points = []
  for m, n in matches:
    if m.distance < LRatio*n.distance:
        good_points.append(m)
  print(len(good_points))
  result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
#  cv2.imshow("result", cv2.resize(result,(800,600)))
  return (good_points, result)

try:
  args = vars(ap.parse_args())
except:
  print('War: Exception in arg parse, using default values')
  args = dict()
  
arg_img = args['image']
arg_pattern = args['pattern']
print("Image is file:"+str(os.path.isfile(arg_img)))
print("Image is directory:"+str(os.path.isdir(arg_img)))
print("Pattern is file:"+str(os.path.isfile(arg_pattern)))
print("Pattern is directory:"+str(os.path.isdir(arg_pattern)))

imlst = list()
pattlst = list()

if os.path.isfile(arg_img):
  imlst.append(arg_img)
elif os.path.isdir(arg_img):
  for im in paths.list_images(arg_img):
    imlst.append(im)
elif not os.path.exists(arg_img):
  print('Err: Path given in image arg not found:'+arg_img)
else:
  print('Err: Unexpected condition with image arg:'+arg_img)

if os.path.isfile(arg_pattern):
  pattlst.append(arg_pattern)
elif os.path.isdir(arg_pattern):
  for im in paths.list_images(arg_pattern):
    pattlst.append(im)
elif not os.path.exists(arg_pattern):
  print('Err: Path given in pattern arg not found:'+arg_pattern)
else:
  print('Error: Unexpected condition with pattern arg:'+arg_pattern)

# In[]
key = 0
result_dict = dict()
imgcnt = 0
for im in imlst:
  imgcnt+=1
  #image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #pattern2 = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
  imname = os.path.split(im)[-1].split('.')[0]
  print("Processing image ({}) {}/{}".format(imname,imgcnt,len(imlst)))
  
  image = cv2.imread(im, cv2.IMREAD_UNCHANGED)
#  image2 = cv2.resize(image, (600,600))
  image2 = image.copy()
  if blur_images:
    image2 = cv2.GaussianBlur(image2,BLER_KERNEL_SIZE,0)
  
  #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
  
  #cv2.imshow("image_gray", imutils.resize(image_gray,width=600))
  #cv2.imshow("pattern_gray", imutils.resize(pattern_gray,width=600))
  pattcnt = 0
  for patt in pattlst:
    pattcnt+=1
    pattname = os.path.split(patt)[-1].split('.')[0]
    print("Processing pattern ({}) {}/{}".format(pattname,pattcnt,len(pattlst)))
    pattern = cv2.imread(patt, cv2.IMREAD_UNCHANGED)
#    pattern2 = cv2.resize(pattern, (600,600))
    pattern2 = pattern.copy()
    pattern2 = convert_rgba2rgb(pattern2)
    if blur_images:
      pattern2 = cv2.GaussianBlur(pattern2,BLER_KERNEL_SIZE,0)
    
    if useMatcher:
      matcher = Matcher()
      (matches, status, vis) = matcher.match([image2, pattern2], ratio=LRatio, showMatches=True, useRootSIFT=useRootSIFT)
#      match_ctr = np.sum(status)
#      (kpsA, featuresA) = matcher.detectAndDescribe(image2, useRootSIFT=useRootSIFT)
#      (kpsB, featuresB) = matcher.detectAndDescribe(pattern2, useRootSIFT=useRootSIFT)
    else:
      matches, vis = matchImages(pattern2, image2, useFlann=useFlann)
    
    if type(matches) == type(None):
      match_ctr = 0
    else:
      match_ctr = len(matches)
      
    if match_ctr >= MIN_MATCH_PTS:
      if len(vis) != 0:
        if os.path.isfile(arg_img) and os.path.isfile(arg_pattern):
          cv2.imshow("matches_{}_{}_({})".format(imname,pattname,match_ctr), cv2.resize(vis, (1200,600)))
        cv2.imwrite("output/{}_{}_{}.png".format(imname,pattname,match_ctr), vis)
      else:
        print("No match found")
        if os.path.isfile(arg_img) and os.path.isfile(arg_pattern):
          cv2.imshow("No matches_{}_{}".format(imname,pattname), np.hstack([cv2.resize(image2, (600,600)), cv2.resize(pattern2[:,:,:3], (600,600))]))
    else:
      print("Insufficient matches found")
      if os.path.isfile(arg_img) and os.path.isfile(arg_pattern):
        cv2.imshow("Insufficient matches_{}_{}_({})".format(imname,pattname,match_ctr), np.hstack([cv2.resize(image2, (600,600)), cv2.resize(pattern2[:,:,:3], (600,600))]))
    
    if os.path.isfile(arg_img) and os.path.isfile(arg_pattern):
      key = cv2.waitKey(0)
      cv2.destroyAllWindows()

    result_dict[imname+'+'+pattname] = match_ctr
    
    if key == 27:
      break

  if key == 27:
      break

if key == 27:
  print('Aborted execution')
else:
  print('Finished processing all images')

# In[]
if os.path.isdir(arg_img) and os.path.isdir(arg_pattern):
  df = pd.DataFrame([result_dict])
  df.to_csv('output/result.csv', index=False)

# In[]
