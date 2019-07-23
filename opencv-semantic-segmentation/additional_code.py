# loop over each of the individual class IDs in the image
for idx, classID in enumerate(np.unique(classMap)):
	# build a binary mask for the current class and then use the mask
	# to visualize all pixels in the image belonging to the class
	print("[INFO] class: {}".format(CLASSES[classID]))
	classMask = (mask == COLORS[classID]).astype("uint8") * 255
	classMask = classMask[:, :, 0]
	classOutput = cv2.bitwise_and(image, image, mask=classMask)
	classMask = np.hstack([image, classOutput])

	# show the output class visualization
	className = "[INFO] class: {} {}/{}".format(CLASSES[classID], idx+1, len(np.unique(classMap)));
	cv2.putText(classMask, className, (5, 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	cv2.imshow("Class Vis", classMask)
	cv2.waitKey(0)