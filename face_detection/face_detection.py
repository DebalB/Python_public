# USAGE
# python face_detection.py --face-cascade cascades/haarcascade_frontalface_default.xml --output output/faces/face_data.txt --input-img avatars/potter.jpg

# import the necessary packages
from __future__ import print_function
from pyimagesearch.face_recognition import FaceDetector
from imutils import encodings
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-i", "--input-img", required=True, help="path to input image file")
args = vars(ap.parse_args())

# initialize the face detector, boolean indicating if we are in capturing mode or not, and
# the bounding box color
fd = FaceDetector(args["face_cascade"])
captureMode = False
color = (0, 255, 0)
saveImages = True

img = cv2.imread(args["input_img"])
img = imutils.resize(img, width=500)
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("input-gray", gray2)
faceRects2 = fd.detect(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
if len(faceRects2) > 0:
	if len(faceRects2) > 1:
		faceRects_sorted = sorted(faceRects2, key=lambda b:(b[2] * b[3]), reverse=True)
		(x3,y3,w3,h3) = faceRects_sorted[0]
		(x4,y4,w4,h4) = faceRects_sorted[1]
		roi = img[y3:y3+h3,x3:x3+w3]
		roi2 = img[y4:y4+h4,x4:x4+w4]
		cv2.imshow("roi-img", roi)
		cv2.imshow("roi-img2", roi2)
	else:
		# sort the bounding boxes, keeping only the largest one
		(x2, y2, w2, h2) = max(faceRects2, key=lambda b:(b[2] * b[3]))
		roi = img[y2:y2+h2,x2:x2+w2]
		cv2.imshow("roi-img", roi)

key = cv2.waitKey(0) & 0xFF
if key == ord("q"):
	cv2.destroyAllWindows()
	exit()

# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)
f = open(args["output"], args["write_mode"])
dirname = os.path.dirname(args["output"])
total = 0

# loop over the frames of the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break

	# resize the frame, convert the frame to grayscale, and detect faces in the frame
	frame = imutils.resize(frame, width=500)
	frame_copy = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))

	# ensure that at least one face was detected
	if len(faceRects) > 0:
		# sort the bounding boxes, keeping only the largest one
		(x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))

		# if we are in capture mode, extract the face ROI, encode it, and write it to file
		if captureMode:
			face = gray[y:y + h, x:x + w].copy(order="C")
			f.write("{}\n".format(encodings.base64_encode_image(face)))
			total += 1

		# draw bounding box on the frame
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
		# 	roi_new = imutils.resize(roi, height=h)
		roi_new = cv2.resize(roi,(h,w))
		frame_copy[y:y+h,x:x+w] = roi_new
		cv2.imshow("Frame_new", frame_copy)

	# show the frame and record if the user presses a key
	if saveImages:
		cv2.imwrite(os.path.join(dirname,str(total)+'.png'),frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `c` key is pressed, then go into capture mode
	if key == ord("c"):
		# if we are not already in capture mode, drop into capture mode
		if not captureMode:
			captureMode = True
			color = (0, 0, 255)

		# otherwise, back out of capture mode
		else:
			captureMode = False
			color = (0, 255, 0)

	# if the `q` key is pressed, break from the loop
	elif key == ord("q"):
		break

# close the output file, cleanup the camera, and close any open windows
print("[INFO] wrote {} frames to file".format(total))
f.close()
camera.release()
cv2.destroyAllWindows()
