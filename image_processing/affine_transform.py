# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:12:49 2019

@author: DEBAL
"""

import cv2
import numpy as np
img = cv2.imread("grid.jpg")
rows, cols, ch = img.shape
cv2.circle(img, (83, 90), 5, (0, 0, 255), -1)
cv2.circle(img, (447, 90), 5, (0, 0, 255), -1)
cv2.circle(img, (83, 472), 5, (0, 0, 255), -1)
pts1 = np.float32([[83, 90], [447, 90], [83, 472]])
pts2 = np.float32([[83, 90], [447, 90], [150, 472]])
pts3 = np.float32([[0, 90], [447, 90], [150, 472]])
#pts2 = np.float32([[0, 0], [447, 90], [150, 472]])
matrix = cv2.getAffineTransform(pts1, pts2)
matrix2 = cv2.getAffineTransform(pts1, pts3)
result = cv2.warpAffine(img, matrix, (cols, rows))
result2= cv2.warpAffine(img, matrix2, (cols, rows))
cv2.imshow("Image", img)
cv2.imshow("Affine transformation", result)
cv2.imshow("Affine transformation-2", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
