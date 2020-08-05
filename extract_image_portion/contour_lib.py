# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:54:21 2019

@author: DEBAL
"""

import cv2
import numpy as np
import statistics

def draw_contours(input_img, contours_list, color=(0,0,255), thickness=1, debug=False):
	output_img = input_img.copy()
	#max_contour = contours_list[0]
	#min_contour = contours_list[-1]
	
	#tmpArea = np.zeros(input_img.shape)
	#cv2.drawContours(tmpArea,max_contour,0,255,5)
	#cv2.drawContours(tmpArea,min_contour,0,255,2)
	#cv2.imshow("tmpArea", tmpArea)
	
	#imAreaOpen0 = np.zeros(image.shape)
	for idx, c_info in enumerate(contours_list):
		cv2.drawContours(output_img,[c_info[0]],0,color,thickness)
		if (debug):
			tmpArea = np.zeros(input_img.shape)
			cv2.drawContours(tmpArea,[c_info[0]],0,color,thickness)
			label = "Count={}/{}, Area={:.2f}".format(idx+1,len(contours_list),c_info[2])
			cv2.putText(tmpArea, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
			cv2.imshow("tmpArea", tmpArea)
			key = cv2.waitKey(0)
			if key == 27:
				debug = False
		
	return output_img

def get_contours(input_img, min_contour_area=10.0, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE):

	contour, hier = cv2.findContours(input_img, mode, method)
	
	contour_info = []
	num_nonzeroarea_cnts = 0
	num_zeroarea_cnts = 0
	for c in contour:
		if (cv2.contourArea(c) >= min_contour_area):
			num_nonzeroarea_cnts += 1
			contour_info.append((
				c,
				cv2.isContourConvex(c),
				cv2.contourArea(c),
			))
		else:
			num_zeroarea_cnts += 1
			
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	return contour_info

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def draw_lines(input_img, contours_list, color=(0,0,255), thickness=1):
	output_img = input_img.copy()
	for idx, c_info in enumerate(contours_list):
		x,y,w,h = cv2.boundingRect(c_info[0])
		rect = cv2.minAreaRect(c_info[0])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(output_img,[box],0,color,thickness)
        #rect_area = h*w
        #print("Rect Area("+str(idx)+")="+str(rect_area))
		#if (rect_area >=100 and rect_area <=1000):
		#	cv2.rectangle(output_img,(x,y),(x+w,y+h),color,thickness)
	return output_img

def draw_circles(input_img, contours_list, color=(0,0,255), thickness=1, debug=False):
	output_img = input_img.copy()
	radius_list = []
	for idx, c_info in enumerate(contours_list):
		((x, y), radius) = cv2.minEnclosingCircle(c_info[0])
		if radius > 5 and radius < 10:
			radius_list.append(radius)

	median_radius = statistics.median(radius_list)
        
	for idx, c_info in enumerate(contours_list):
		((cx,cy), (width, height), angle) = cv2.minAreaRect(c_info[0])
		aspect_ratio = min(width, height)/max(width, height)
		((x, y), radius) = cv2.minEnclosingCircle(c_info[0])
		#M = cv2.moments(c_info[0])
		#center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if radius > 5 and radius < 10 and aspect_ratio > 0.80:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			draw_radius = max(radius, median_radius)
			cv2.circle(output_img, (int(x), int(y)), int(draw_radius),
				color, thickness)
		elif radius >= 10 and radius < 50 and aspect_ratio > 0.50:
			# likely to be few balls close together
			cv2.drawContours(output_img,[c_info[0]],0,color,thickness)
		else:
			# contour is either too small or too big or not balls and should be ignored
			pass

			#cv2.circle(output_img, center, 5, (0, 0, 255), -1)
		if (debug):
			tmpArea = np.zeros(input_img.shape)
			cv2.drawContours(tmpArea,[c_info[0]],0,color,thickness)
			label = "Count={}/{}, Area={:.2f}, Radius={:.2f}".format(idx+1,len(contours_list),c_info[2],radius)
			cv2.putText(tmpArea, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			label2 = "MinAreaRect Cen=({:.0f},{:.0f}), W={:.2f}, H={:.2f}, Ang={:.2f}, AR={:.2f}".format(cx,cy,width,height,angle,aspect_ratio)
			cv2.putText(tmpArea, label2, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.imshow("tmpArea", tmpArea)
			key = cv2.waitKey(0)
			if key == 27:
				debug = False
	
	return output_img
  
def detect_lines(input_img, threshold=200, color=(0,0,255), thickness=1, debug=False):
	output_img = input_img.copy()
	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.bilateralFilter(gray, 5, 15, 15)
	#canny_img = cv2.Canny(blurred, 30, 150)
	autocanny = auto_canny(blurred)
	(h, w) = input_img.shape[:2]
	x1_top, x1_bottom, x1_left, x2_left = 0,0,0,0
	y1_top, y1_left, y2_top, y1_right = 0,0,0,0
	x2_top, x2_bottom, x1_right, x2_right = w,w,w,w
	y1_bottom, y2_bottom, y2_left, y2_right = h,h,h,h

	lines = cv2.HoughLines(autocanny,1,np.pi/180,threshold)
	for elem in lines:
		for rho,theta in elem:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			a_new = np.float("{:.2f}".format(abs(a)))
			b_new = np.float("{:.2f}".format(abs(b)))
			if ((a_new == 1.00) and (b_new == 0.00)):
				# vertical line
				#cv2.line(output_img,(x1,y1),(x2,y2),color,thickness)
				# find out if this vertical line is closer to left or right edge
				if (max(x1,x2) < (w/2)):
					# closer to left edge
					if (max(x1,x2) > max(x1_left, x2_left)):
						x1_left, x2_left = max(x1,x2), max(x1,x2)
						y1_left, y2_left = y1, y2
				else:
					#closer to right edge
					if (min(x1,x2) < min(x1_right, x2_right)):
						x1_right, x2_right = min(x1,x2), min(x1,x2)
						y1_right, y2_right = y1, y2
			elif ((a_new == 0.00) and (b_new == 1.00)):
				# horizontal line
				#cv2.line(output_img,(x1,y1),(x2,y2),color,thickness)
				# find out if this horizontal line is closer to top or bottom edge
				if (max(y1,y2) < (h/2)):
					# closer to top edge
					if (max(y1,y2) > max(y1_top, y2_top)):
						y1_top, y2_top = max(y1,y2), max(y1,y2)
						x1_top, x2_top = x1, x2
				else:
					#closer to bottom edge
					if (min(y1,y2) < min(y1_bottom, y2_bottom)):
						y1_bottom, y2_bottom = min(y1,y2), min(y1,y2)
						x1_bottom, x2_bottom = x1, x2
				
			if (debug):
				tmpArea = np.zeros(input_img.shape)
				cv2.line(tmpArea,(x1,y1),(x2,y2),color,thickness)
				label = "rho={}, sintheta={:.2f}, costheta={:.2f}, x1={} y1={}, x2={}, y2={}".format(rho,b,a,x1,y1,x2,y2)
				cv2.putText(tmpArea, label, (100, 200),				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
				cv2.imshow("tmpArea", tmpArea)
				key = cv2.waitKey(0)
				if key == 27:
					debug = False
	
	cv2.line(output_img,(x1_top,y1_top),(x2_top,y2_top),(255,0,0),thickness)
	cv2.line(output_img,(x1_bottom,y1_bottom),(x2_bottom,y2_bottom),(255,0,0),thickness)
	cv2.line(output_img,(x1_left,y1_left),(x2_left,y2_left),(255,0,0),thickness)
	cv2.line(output_img,(x1_right,y1_right),(x2_right,y2_right),(255,0,0),thickness)
	
	return output_img;

def detect_circles(input_img, min_dst=10, minRadius=1, maxRadius=100):
	# detect circles in the image
	output = input_img.copy()
	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, min_dst, minRadius, maxRadius)
	 
	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
	 
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	 
		# show the output image
		#cv2.imshow("circles", np.hstack([image, output]))
		cv2.imshow("circles", output)


def get_image_mask(input_img, hsv_lower, hsv_upper):
	# define the lower and upper boundaries of the "green"
	# ball in the HSV color space, then initialize the
	# list of tracked points
	#hsv_lower = (67, 0, 0)
	#hsv_upper = (86, 255, 255)
	
	hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	img_mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
	#cv2.imshow("img_mask", img_mask)
	return img_mask