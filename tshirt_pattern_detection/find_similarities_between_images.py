# Usage:
#--image images/TrocknerBilder/temp1.png --pattern images/TrocknerDruckdaten/32BZ5BXY.png
#--image images/TrocknerBilder/temp2.png --pattern images/TrocknerDruckdaten/XT2974JT.png
#--image images/TrocknerBilder/temp3.png --pattern images/TrocknerDruckdaten/M83QYD29.png
#--image images/TrocknerBilder/temp4.png --pattern images/TrocknerDruckdaten/4VY1YHI4.png
#--image images/TrocknerBilder/temp1.png --pattern images/TrocknerDruckdaten/4VY1YHI4.png
#--image images/TrocknerBilder/temp1.png --pattern images/TrocknerDruckdaten/GREMSV67.png
#--image images/TrocknerBilder/temp1.png --pattern images/TrocknerDruckdaten/XT2974JT.png
#--image images/TrocknerBilder/temp1.png --pattern images/TrocknerDruckdaten
#--image images/TrocknerBilder --pattern images/TrocknerDruckdaten

import cv2
import numpy as np

#original = cv2.imread("images/TrocknerBilder/temp1.png")
#image_to_compare = cv2.imread("images/TrocknerDruckdaten/32BZ5BXY.png")

#original = cv2.imread("images/TrocknerBilder/temp2.png")
#image_to_compare = cv2.imread("images/TrocknerDruckdaten/XT2974JT.png")

original = cv2.imread("images/TrocknerBilder/temp3.png")
image_to_compare = cv2.imread("images/TrocknerDruckdaten/M83QYD29.png")

#original = cv2.imread("images/TrocknerBilder/temp4.png")
#image_to_compare = cv2.imread("images/TrocknerDruckdaten/4VY1YHI4.png")

# 1) Check if 2 images are equals
if original.shape == image_to_compare.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely Equal")
    else:
        print("The images are NOT equal")
else:
    print("The images have different sizes")
#    original = cv2.resize(original, (1200,1200))
#    image_to_compare = cv2.resize(image_to_compare, (1200, 1200))


# 2) Check for similarities between the 2 images

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
ratio = 0.6
for m, n in matches:
	if m.distance < ratio*n.distance:
		good_points.append(m)
print(len(good_points))
result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

cv2.imshow("result", cv2.resize(result,(800,600)))
cv2.imshow("Original", cv2.resize(original,(800,600)))
cv2.imshow("Duplicate", cv2.resize(image_to_compare,(800,600)))

number_keypoints = 0
if len(kp_1) >= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)
percentage_similarity = len(good_points) / number_keypoints * 100
print("Similarity: " + str(int(percentage_similarity)) + "%\n")

cv2.waitKey(0)
cv2.destroyAllWindows()
