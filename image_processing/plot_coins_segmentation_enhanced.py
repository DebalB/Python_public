# USAGE
# python plot_coins_segmentation_enhanced.py --image ../images/DSC_0369.JPG

import argparse
import cv2
import numpy as np
import imutils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data
from skimage.exposure import histogram

from skimage import color
from skimage import io

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
image = imutils.resize(image, width=384)
cv2.imshow("image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_img1 = color.rgb2gray(io.imread(args["image"]))
gray_img2 = io.imread(args["image"], as_gray=True)

coins = data.coins()
#coins = gray.copy()
#coins_gray = color.rgb2gray(coins)
cv2.imshow("coins", coins)

hist, hist_centers = histogram(coins)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(coins, cmap=plt.cm.gray)
#axes[0].imshow(coins, cmap=plt.cm.gist_rainbow)
axes[0].axis('off')
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')

######################################################################
#
# Thresholding
# ============
#
# A simple way to segment the coins is to choose a threshold based on the
# histogram of gray values. Unfortunately, thresholding this image gives a
# binary image that either misses significant parts of the coins or merges
# parts of the background with the coins:

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axes[0].imshow(coins > 100, cmap=plt.cm.gray)
axes[0].set_title('coins > 100')

axes[1].imshow(coins > 150, cmap=plt.cm.gray)
axes[1].set_title('coins > 150')

for a in axes:
    a.axis('off')

plt.tight_layout()

######################################################################
# Edge-based segmentation
# =======================
#
# Next, we try to delineate the contours of the coins using edge-based
# segmentation. To do this, we first get the edges of features using the
# Canny edge-detector.

from skimage.feature import canny

#edges = canny(coins)
edges = canny(coins)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(edges, cmap=plt.cm.gray)
ax.set_title('Canny detector')
ax.axis('off')

######################################################################
# These contours are then filled using mathematical morphology.

from scipy import ndimage as ndi

fill_coins = ndi.binary_fill_holes(edges)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_coins, cmap=plt.cm.gray)
ax.set_title('filling the holes')
ax.axis('off')


######################################################################
# Small spurious objects are easily removed by setting a minimum size for
# valid objects.

from skimage import morphology

coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(coins_cleaned, cmap=plt.cm.gray)
ax.set_title('removing small objects')
ax.axis('off')

######################################################################
# However, this method is not very robust, since contours that are not
# perfectly closed are not filled correctly, as is the case for one unfilled
# coin above.
#
# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

from skimage.filters import sobel

elevation_map = sobel(coins)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(elevation_map, cmap=plt.cm.gray)
ax.set_title('elevation map')
ax.axis('off')

######################################################################
# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.nipy_spectral)
ax.set_title('markers')
ax.axis('off')

######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:

segmentation = morphology.watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation, cmap=plt.cm.gray)
ax.set_title('segmentation')
ax.axis('off')

######################################################################
# This last method works even better, and the coins can be segmented and
# labeled individually.

from skimage.color import label2rgb

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_coins, image=coins)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(coins, cmap=plt.cm.gray)
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay)

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close("all")
