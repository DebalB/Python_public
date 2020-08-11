# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:04:08 2020

@author: DEBAL
"""

import numpy as np
import cv2
from collections import namedtuple


def find_rects(activations):
    Rect = namedtuple('Rect', 'x1 y1 x2 y2')
    rects_list = []
    # Dilate and erode the activations to remove grid-like artifacts
    kernel = np.ones((5, 5), np.uint8)
    activations = cv2.dilate(activations, kernel=kernel)
    activations = cv2.erode(activations, kernel=kernel)

    # Binarize the activations
    _, activations = cv2.threshold(activations, 0.25, 1, type=cv2.THRESH_BINARY)
    activations = activations.astype(np.uint8).copy()

    # Find the countour of the binary blob
    contours, _ = cv2.findContours(activations.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
      # Find bounding box around the object.
      rect = cv2.boundingRect(cnt)
      rects_list.append((Rect(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])))
  
    return rects_list


def normalize(activations):
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations
