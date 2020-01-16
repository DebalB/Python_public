# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5
# python object_size.py --image images/picsmeasure/sample01.jpg --width 3.5

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def get_ref_obj_cnts(image, hsv_mask, minArea=10.0, debug=False):
  mask_low, mask_high = hsv_mask
  thresh = cv2.inRange(image, mask_low, mask_high)
  
  if debug:
    cv2.imshow("thresh", imutils.resize(thresh, height=600))
  
  cnt_list_temp = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnt_list_temp = imutils.grab_contours(cnt_list_temp)
  cnt_list_final = list()
  for cnt in cnt_list_temp:
    area = cv2.contourArea(cnt)
    if  area >= minArea:
      cnt_list_final.append(cnt)
  return cnt_list_final

def get_ref_obj_pts(ref_cnts_list):
  pts = list()
  for cnt in ref_cnts_list:
    box = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(box))
    pts.append(box)
  return pts

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

DEBUG = False
pixelsPerMetric = None

actual_dims = {"coin_10": (2.7, 2.7),
            "coin_5": (2.3, 2.3),
            "key": (7.4, 3.3),
            "card": (9.1, 5.5),
            "case": (16.4,6.0)}
print("actual object dimensions:")
[print("{}:{},{}".format(item,length,breadth)) for (item,(length,breadth)) in actual_dims.items()]

# define reference obj HSV values
l_h = 106
l_s = 42
l_v = 0
u_h = 120
u_s = 255
u_v = 255
ref_obj_mask = (np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))

image_hsv = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2HSV)
ref_obj_cnt_lst = get_ref_obj_cnts(image_hsv, ref_obj_mask, minArea=100.0, debug=DEBUG)
ref_obj_pts_lst = get_ref_obj_pts(ref_obj_cnt_lst)

if DEBUG:
  ref_obj_img = np.zeros(image_hsv.shape, dtype=np.uint8)
  for pts in ref_obj_pts_lst:
    cv2.drawContours(ref_obj_img, [pts], -1, (255,255,255), 5)
  cv2.imshow("ref obj", imutils.resize(ref_obj_img, height=600))

if len(ref_obj_pts_lst) == 1:
  # There should be only one reference object match in the image in order to calculate the refernce dimensions accurately
  (tl, tr, br, bl) = perspective.order_points(ref_obj_pts_lst[0])
  (tltrX, tltrY) = midpoint(tl, tr)
  (blbrX, blbrY) = midpoint(bl, br)

  # compute the midpoint between the top-left and bottom-left points,
  # followed by the midpoint between the top-right and bottom-right
  (tlblX, tlblY) = midpoint(tl, bl)
  (trbrX, trbrY) = midpoint(tr, br)

  # compute the Euclidean distance between the midpoints
  dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
  dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
  
  px_width = max(dA, dB)
  
  if pixelsPerMetric is None:
    pixelsPerMetric = px_width / actual_dims['card'][0]
    
  if DEBUG:
    print("Calculated Ref Obj values:")
    print("dA:", dA)
    print("dB:", dB)
    print("pixelsPerMetric:", pixelsPerMetric)
  
else:
  # In case reference object is not found in the image, we will calculate it as per the supplied argument later
  pixelsPerMetric = None

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

orig = image.copy()
idx = 0
# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	# compute the rotated bounding box of the contour
# 	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and bottom-left points,
	# followed by the midpoint between the top-right and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	
	print("Obj-{}: dimA,dimB=({},{})".format(idx+1,dimA,dimB))

	# draw the object sizes on the image
	cv2.putText(orig, "Obj-{}".format(idx+1),
		(int(tltrX-400), int(tltrY-50)), cv2.FONT_HERSHEY_SIMPLEX,
		4.0, (0, 0, 255), 8)
	cv2.putText(orig, "{:.1f}".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		4.0, (255, 0, 255), 6)
	cv2.putText(orig, "{:.1f}".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		4.0, (255, 0, 255), 6)

	# show the output image
	cv2.imshow("Image", imutils.resize(orig,height=600))
	cv2.waitKey(0)
	idx+=1
    
cv2. destroyAllWindows()