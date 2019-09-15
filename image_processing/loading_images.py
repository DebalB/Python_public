import cv2
image = cv2.imread("red_panda.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray panda", gray_image)
cv2.imshow("Red panda", image)

cv2.imwrite("gray_panda.jpg", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


