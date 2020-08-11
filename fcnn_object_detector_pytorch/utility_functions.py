# -*- coding: utf-8 -*-
"""
Code based on "https://github.com/spmallick/learnopencv/tree/master/PyTorch-Receptive-Field-With-Backprop"
"""

import numpy as np
import cv2
from collections import namedtuple
import torch
import torch.nn as nn
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18

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


def backprop_receptive_field(image, predicted_class, scoremap, use_max_activation=True,max_loc=None,useGpu=False):
    model = FullyConvolutionalResnet18()
    
    model = model.train()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05) # inference overflows with ones
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except:
            pass

        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

    input = torch.ones_like(image, requires_grad=True)
    
    if useGpu == True and torch.cuda.is_available() == True:
      # print('moving to GPU {model, input, scoremap, predicted_class}')
      model = model.cuda()
      input = input.cuda()
      scoremap = scoremap.cuda()
      predicted_class = predicted_class.cuda()

    out = model(input)
    grad = torch.zeros_like(out, requires_grad=True)

    if not use_max_activation:
      grad[0, predicted_class] = scoremap
    elif max_loc != None:
      # print('Coords provided for max activation:', max_loc[0], max_loc[1])
      grad[0, 0, max_loc[0], max_loc[1]] = 1
    else:
      scoremap_max_row_values, max_row_id = torch.max(scoremap, dim=1)
      _, max_col_id = torch.max(scoremap_max_row_values, dim=1)
      max_row_id = max_row_id[0, max_col_id]
      # print('Coords of the max activation:', max_row_id.item(), max_col_id.item())
      grad[0, 0, max_row_id, max_col_id] = 1
    
    out.backward(gradient=grad)
    gradient_of_input = input.grad[0, 0].cpu().data.numpy()
    # gradient_of_input = gradient_of_input / np.amax(gradient_of_input)

    return gradient_of_input
  