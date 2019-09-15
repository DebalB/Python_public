# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:54:21 2019

@author: DEBAL
"""
# USAGE
# python extract_image_portion.py --image ../images/Image1.png
# --image images/image13.png

## Image processing example
#import os
import cv2
import numpy as np
import contour_lib
import argparse
import sys
#from matplotlib import pyplot as plt
#%matplotlib inline

#MIN_CONTOUR_AREA = 14000
MIN_CONTOUR_AREA = 7700
RESIZE_IMAGE = False
DEBUG = False

def get_map_segments(contours_list, min_contour_area=MIN_CONTOUR_AREA, debug=False):

  contour_list_final = []
  top_left_contour = [[0,0]]
  top_right_contour = [[0,0]]
  bottom_left_contour = [[0,0]]
  bottom_right_contour = [[0,0]]
  result_bitmap = 0
  
  for idx, c in enumerate(contours_list):
    cx,cy,cw,ch = cv2.boundingRect(c[0])
    car = min(cw,ch)/max(cw,ch)
    #peri = cv2.arcLength(c[0], True)
    #approx = cv2.approxPolyDP(c[0], 0.02 * peri, True)
    #length = len(approx)
    tmpArea = np.zeros(clone.shape)
    cv2.drawContours(tmpArea,[c[0]],0,(255,255,255),2)
    length = len(c[0])
    #cv2.fillPoly(tmpArea,[c[0]],(255,255,255))
    #cv2.drawContours(tmpArea,[approx],0,(255,255,255),2)
    #cv2.fillPoly(tmpArea,[approx],(255,255,255))
    #cv2.polylines(tmpArea,[approx],True,(255,255,255), 2)
    M = cv2.moments(c[0])
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    label = "Cnt={}/{}, Area={:.2f}, Len={}, Cen={}".format(idx+1,len(contours_list),c[2],length, center)
    cv2.putText(tmpArea, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    label2 = "(X,Y)=({},{}), W={:.2f}, H={:.2f}, AR={:.2f}".format(cx,cy,cw,ch,car)
    cv2.putText(tmpArea, label2, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    if debug:
      cv2.imshow("tmpArea", tmpArea)
    
    # exlcude rectangular and small shapes
    if car > 0.80 and c[2] > MIN_CONTOUR_AREA:
      #check if contour is top left
      if (center[0] < img_centre_x) and (center[1] < img_centre_y):
        if c[2] > top_left_contour[0][1]:
          top_left_contour[0] = [c[0], c[2]]
          print("found top left image")
          result_bitmap = np.bitwise_or(result_bitmap, 1)
      #check if contour is top right
      elif (center[0] > img_centre_x) and (center[1] < img_centre_y):
        if c[2] > top_right_contour[0][1]:
          top_right_contour[0] = [c[0], c[2]]
          print("found top right image")
          result_bitmap = np.bitwise_or(result_bitmap, 2)
      #check if contour is bottom left
      elif (center[0] < img_centre_x) and (center[1] > img_centre_y):
        if c[2] > bottom_left_contour[0][1]:
          bottom_left_contour[0] = [c[0], c[2]]
          print("found bottom left image")
          result_bitmap = np.bitwise_or(result_bitmap, 4)
      #check if contour is bottom right
      elif (center[0] > img_centre_x) and (center[1] > img_centre_y):
        if c[2] > bottom_right_contour[0][1]:
          bottom_right_contour[0] = [c[0], c[2]]
          print("found bottom right image")
          result_bitmap = np.bitwise_or(result_bitmap, 8)
      # invalid condition
      else:
        print("Err: Invalid condition")
      
    if debug:
      key = cv2.waitKey(0)
      if key == 27:
        debug=False
  
  contour_list_final.append([top_left_contour, top_right_contour, bottom_left_contour, bottom_right_contour])
  return (contour_list_final, result_bitmap)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())
filename = args["image"]

cv2.destroyAllWindows()

img = cv2.imread(filename)
(H, W) = img.shape[:2]
print("Image Shape", (H, W))
if RESIZE_IMAGE:
  img = cv2.resize(img, (800,600))
  (H, W) = img.shape[:2]
  print("Image Shape New", img.shape)
clone = img.copy()

img_centre_x = W//2
img_centre_y = H//2
print("Image centre = ({},{})".format(img_centre_x, img_centre_y))

cv2.imshow("Original", img)
if len(img.shape) == 3:
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale", img)

blurred = cv2.GaussianBlur(img, (11, 11), 0)
#cv2.imshow("Blurred", blurred)
#laplacian = cv2.Laplacian(blurred,cv2.CV_64F)
#cv2.imshow("laplacian", laplacian)
canny = cv2.Canny(blurred, 30, 150)
cv2.imshow("Canny", canny)

dilateSize = 3
#dilateSize = 15
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateSize,dilateSize))
#canny_morphed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
#canny_morphed = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
canny_morphed = canny
canny_morphed  = cv2.dilate(canny_morphed,kernel,iterations=5)
#cv2.imshow("canny_morphed1", canny_morphed)
canny_morphed  = cv2.erode(canny_morphed,kernel,iterations=5)
cv2.imshow("canny_morphed2", canny_morphed)

contours_list = contour_lib.get_contours(canny_morphed, min_contour_area=1500.0)
#contour_img = contour_lib.draw_contours(clone, contours_list, color=(0,0,255), thickness=1, debug=True)
#cv2.imshow("contour_img", contour_img)

contour_list_final, result_bitmap =  get_map_segments(contours_list, debug=DEBUG)
print("result_bitmap="+str(result_bitmap))
if (result_bitmap != 15):
  print("Error: Extracted "+str((result_bitmap+1)//2)+" segments")
  sys.exit()
else:
  print("Successfully extracted all 4 segments")

tmpAreaBlack = np.zeros(clone.shape)

for c in contour_list_final[0]:
  #print("No of points in contour= "+str(len(c[0])))
  cv2.drawContours(tmpAreaBlack,c[0],0,(255,255,255),cv2.FILLED)
  cv2.imshow("tmpAreaBlack", tmpAreaBlack)
  
#mask = (tmpArea != 0)
#mask = tmpArea[:,:,0].astype(np.uint8)
mask = tmpAreaBlack[:,:,0].astype("uint8")
new_img_blk_bg = cv2.bitwise_and(clone, clone, mask=mask)
cv2.imshow("Image Black Background", new_img_blk_bg)

tmpAreaWhite = np.ones(clone.shape)*255
for c in contour_list_final[0]:
  #print("No of points in contour= "+str(len(c[0])))
  cv2.drawContours(tmpAreaWhite,c[0],0,(0,0,0),cv2.FILLED)
cv2.imshow("tmpAreaWhite", tmpAreaWhite)
mask = tmpAreaWhite[:,:,0].astype("uint8")
new_img_white = cv2.add(new_img_blk_bg, tmpAreaWhite.astype("uint8"))
cv2.imshow("Image White Background", new_img_white)

#new_img_white2 = np.where(new_img_blk_bg[:,:,:] == 0, 255, new_img_blk_bg[:,:,:])
#cv2.imshow("Image White Background-2", new_img_white2)
##check if images are not identical
#print(np.where(new_img_white2 != new_img_white))

cv2.waitKey(0)
print("Closing all windows")
cv2.destroyAllWindows()