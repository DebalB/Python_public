# USAGE
# python pool_table_detection.py --image images/image1.png
# --image images/image13.png

# import the necessary packages
import argparse
import cv2
import os
import imutils
#import numpy as np
#from scipy import ndimage as ndi
#from skimage.feature import canny
#import statistics
import sys
import contour_lib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())

MIN_BALL_AREA = 300

# check if image file exists
if not (os.path.exists(args["image"])):
  print("[ERR] Image path \"{}\" does not exist".format(args["image"]));
  sys.exit();
# load the input image and grab its dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
image = imutils.resize(image, width=800)

# convert the image to grayscale, blur it, and perform Canny
# edge detection
print("[INFO] performing Canny edge detection...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = gray
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv.bilateralFilter() is highly effective in noise removal while keeping edges sharp
blurred = cv2.bilateralFilter(gray, 5, 15, 15)
canny_img = cv2.Canny(blurred, 30, 150)
autocanny = contour_lib.auto_canny(blurred)

cv2.imshow("Input", image)
#cv2.imshow("Gray", gray)
#cv2.imshow("blurred", blurred)
#cv2.imshow("Canny", canny_img)
#cv2.imshow("autocanny", autocanny)

#edges = canny(gray/255.)
#fill_holes = ndi.binary_fill_holes(edges)
#cv2.imshow("fill_holes", fill_holes)

balls_mask = contour_lib.get_image_mask(image, (67, 0, 0), (86, 255, 255))
table_edges_mask = contour_lib.get_image_mask(image, (67, 0, 116), (86, 255, 255))

cv2.imshow("balls_mask", balls_mask)
cv2.imshow("table_edges_mask", table_edges_mask)

balls_mask_new = balls_mask.copy()
dilateSize = 3
#dilateSize = 15
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateSize,dilateSize))
#balls_mask_new = cv2.morphologyEx(balls_mask, cv2.MORPH_CLOSE, kernel)
#balls_mask_new = cv2.morphologyEx(balls_mask, cv2.MORPH_OPEN, kernel)
balls_mask_new = cv2.dilate(balls_mask_new,kernel,iterations=2)
balls_mask_new = cv2.erode(balls_mask_new,kernel,iterations=2)
#cv2.imshow("balls_mask_new", balls_mask_new)

# Table would be at least half the whole image size
min_table_area = 0.5 * image.shape[0] * image.shape[1]
table_contours = contour_lib.get_contours(balls_mask_new, min_contour_area=image.shape[0])
table_image1 = contour_lib.draw_contours(image, table_contours,(0,0,255),2, debug=False)
#cv2.imshow("table_image1", table_image1)
Tx,Ty,Tw,Th = cv2.boundingRect(table_contours[0][0])
table_image_short = image[Ty:Ty+Th,Tx:Tx+Tw,:]
cv2.imshow("table_image_short", table_image_short)
balls_mask_new = balls_mask_new[Ty:Ty+Th,Tx:Tx+Tw]
image_clone = image.copy()
image = table_image_short

#ball_contours = contour_lib.get_contours(balls_mask)
ball_contours = contour_lib.get_contours(balls_mask_new)
ball_contours_final = list()
num_balls_found = 0
for c in ball_contours:
		if (c[2] <= MIN_BALL_AREA):
			num_balls_found += 1
			ball_contours_final.append(c)
#print("Found "+str(num_balls_found)+" balls in the image")
			
#table_contours = contour_lib.get_contours(table_edges_mask)
#table_contours_final = list()
#num_table_edges_found = 0
#for c in table_contours:
#		if (c[2] > 50):
#			num_table_edges_found += 1
#			table_contours_final.append(c)
#print("Found "+str(num_table_edges_found)+" table edges in the image")
ball_image1 = contour_lib.draw_contours(image, ball_contours,(0,0,255),2, debug=False)
#cv2.imshow("balls contours", ball_image1)
ball_image = contour_lib.draw_circles(image, ball_contours_final,(255,0,0),-1, debug=False)
#cv2.imshow("balls", ball_image)

#table_edges_image1 = contour_lib.draw_contours(image, table_contours_final,(0,0,255),2, debug=False)
#table_edges_image2 = contour_lib.draw_table_edges(image, table_contours_final,(0,0,255),2)

table_edges_image3 = contour_lib.detect_lines(ball_image, threshold=100, thickness=2, debug=False)

#cv2.fillPoly(imAreaOpen0, max_contour[0], (255))

#cv2.imshow("table edges1", table_edges_image1)
#cv2.imshow("table edges2", table_edges_image2)
#cv2.imshow("table edges3", table_edges_image3)
final_image = image_clone.copy()
final_image[Ty:Ty+Th,Tx:Tx+Tw,:] = table_edges_image3
cv2.imshow("table edges & balls", final_image)
filename = args["image"].split('.')
cv2.imwrite(filename[0]+"_processed."+filename[1],final_image)

print("Finished processing image")
cv2.waitKey(0)
cv2.destroyAllWindows()


