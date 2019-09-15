import cv2
import numpy as np

image = cv2.imread("red_panda.jpg")
rows, cols, ch = image.shape

roi = image[100: 280, 150: 320]

# Print a specific pixel of the image
# We're printing the value in the position 175, 300 
print(image[175, 300])
# Change the value of a pixe
# We are going to assign a new value
image[250, 180] = (255, 0, 0)

cv2.imshow("Panda", image)
cv2.imshow("Roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()