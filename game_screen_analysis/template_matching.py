# Usage
#-i GameScreen1.jpg -t template1.jpg --detector frozen_east_text_detection.pb --padding 0.4 -c 70 -m 0.8
#-i GameScreen2.jpg -t template1.jpg --detector frozen_east_text_detection.pb --padding 0.4 -c 70 -m 0.9
#-i GameScreen2.jpg -t template2.jpg --detector frozen_east_text_detection.pb --padding 0.4 -c 70 -m 0.9
#-i GameScreen1.jpg -t template2.jpg --detector frozen_east_text_detection.pb --padding 0.4 -c 70 -m 0.9

import cv2
import imutils
import argparse
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

# DEBUG = True
DEBUG = False
Text_Threshold = 170

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", type=str, required=True, help="path to input image")
ap.add_argument("-t","--template", type=str, required=True, help="path to input template")
ap.add_argument("-c","--confidence", type=int, default=80.0, help="% confidence level for match")
ap.add_argument("-detector","--detector", type=str, help="path to input EAST text detector")
ap.add_argument("-m","--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w","--width", type=int, default=320, help="nearest multiple of 32 for resized width")
ap.add_argument("-e","--height", type=int, default=320, help="nearest multiple of 32 for resized height")
ap.add_argument("-p","--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")

args = vars(ap.parse_args())

# temporary workaround for Tesseract path issue
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR5\\tesseract.exe'

def decode_predictions(scores, geometry):
  # grab the number of rows and columns from the scores volume, then
  # initialize our set of bounding box rectangles and corresponding
  # confidence scores
  (numRows, numCols) = scores.shape[2:4]
  rects = []
  confidences = []

  # loop over the number of rows
  for y in range(0, numRows):
    # extract the scores (probabilities), followed by the
    # geometrical data used to derive potential bounding box
    # coordinates that surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over the number of columns
    for x in range(0, numCols):
      # if our score does not have sufficient probability,
      # ignore it
      if scoresData[x] < args["min_confidence"]:
          continue

      # compute the offset factor as our resulting feature
      # maps will be 4x smaller than the input image
      (offsetX, offsetY) = (x * 4.0, y * 4.0)

      # extract the rotation angle for the prediction and
      # then compute the sin and cosine
      angle = anglesData[x]
      cos = np.cos(angle)
      sin = np.sin(angle)

      # use the geometry volume to derive the width and height
      # of the bounding box
      h = xData0[x] + xData2[x]
      w = xData1[x] + xData3[x]

      # compute both the starting and ending (x, y)-coordinates
      # for the text prediction bounding box
      endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
      endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
      startX = int(endX - w)
      startY = int(endY - h)

      # add the bounding box coordinates and probability score
      # to our respective lists
      rects.append((startX, startY, endX, endY))
      confidences.append(scoresData[x])

  # return a tuple of the bounding boxes and associated confidences
  return (rects, confidences)

def match_image_hsv(full_image,mask,template,net,layerNames):
  clone = full_image.copy()
  ih,iw,ic = full_image.shape
  th,tw,tc = template.shape
  num_bins = 16
  
  #process template image
  template_blur = cv2.GaussianBlur(template,(199,199),0)
  template_hsv = cv2.cvtColor(template_blur, cv2.COLOR_BGR2HSV)
  # template_hist = cv2.calcHist(template_hsv,[0],None,[180],[0,180])
  template_hist = cv2.calcHist(template_hsv,[0,1,2],None,[num_bins,num_bins,num_bins],[0,180,0,256,0,256])
  # normalize the histogram
  template_hist /= template_hist.sum()
  # plt.figure()
  # plt.plot(template_hist)
  # plt.title("Template Histogram")
  # replace the top left of template hist with max values of hist
  template_hist_h = cv2.calcHist(template_hsv[:,:,0],[0],None,[180],[0,180])
  template_hist_s = cv2.calcHist(template_hsv[:,:,1],[0],None,[256],[0,256])
  template_hist_v = cv2.calcHist(template_hsv[:,:,2],[0],None,[256],[0,256])
  max_h = np.argmax(template_hist_h)
  max_s = np.argmax(template_hist_s)
  max_v = np.argmax(template_hist_v)
  template_hsv[:45,:45,0] = max_h
  template_hsv[:45,:45,1] = max_s
  template_hsv[:45,:45,2] = max_v
  if DEBUG:
    cv2.imshow("template_hsv", imutils.resize(template_hsv, width=400))
  
  cntlst_mask = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cntlst_mask = imutils.grab_contours(cntlst_mask)
  
  if len(cntlst_mask):
    print("No confident matches found")
  else:
    print("Confident Matches found")
    
  for cntidx, cnt in enumerate(cntlst_mask):
    cv2.imshow("input image", imutils.resize(full_image, width=800))
    cv2.imshow("input template", imutils.resize(template, width=400))

    (x1,y1,w,h) = cv2.boundingRect(cnt)
    (x2,y2) = (x1+w,y1+h)
    print("Cnt-{} Box=({},{},{},{})".format(cntidx+1,x1,y1,x2,y2))
    
  # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
  # if DEBUG:
  #   print("Maxval,MaxLoc={:.3},({},{})".format(maxVal,maxLoc[0],maxLoc[1]))
  
  # if maxVal >= args["confidence"]/100:
    # print("Confident Match found")
    # cv2.rectangle(clone, maxLoc, (maxLoc[0] + tw, maxLoc[1] + th), (0, 255, 0), 3)
    
    # obj_roi = image[maxLoc[1]:maxLoc[1]+th, maxLoc[0]:iw//2,:]
    # name_roi = image[maxLoc[1]:maxLoc[1]+(th//2), maxLoc[0]+tw:(iw//2),:]
    # cost_roi = image[maxLoc[1]+(th//2):maxLoc[1]+th, maxLoc[0]+tw:(iw//2),:]
    
    if x1 >= iw//2:
      print("Skipping incorrect match ({},{},{},{})".format(x1,y1,x2,y2))
      continue
    
    obj_roi = full_image[y1:y1+th, x1:iw//2,:]
    name_roi = full_image[y1:y1+(th//2), x1+tw:(iw//2),:]
    cost_roi = full_image[y1+(th//2):y1+th, x1+tw:(iw//2),:]
    obj_roi_short = full_image[y1:y1+th, x1:x1+tw,:]
    if DEBUG:
      cv2.imshow("match roi", imutils.resize(obj_roi, width=400))
      cv2.imshow("name roi", imutils.resize(name_roi, width=400))
      cv2.imshow("cost roi", imutils.resize(cost_roi, width=400))
      cv2.imshow("obj_roi_short", imutils.resize(obj_roi_short, width=400))
      
    # check for match between ROI and template
    # start histogram calculation and comparison
    roi_blur = cv2.GaussianBlur(obj_roi_short,(199,199),0)
    roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)    
    # roi_hist = cv2.calcHist(roi_hsv,[0],None,[180],[0,180])
    roi_hist = cv2.calcHist(roi_hsv,[0,1,2],None,[num_bins,num_bins,num_bins],[0,180,0,256,0,256])
    # normalize the histogram
    roi_hist /= roi_hist.sum()
    # plt.figure()
    # plt.plot(roi_hist)
    # plt.title("ROI Histogram")
    # replace the top left of roi hist with max values from hist
    roi_hist_h = cv2.calcHist(roi_hsv[:,:,0],[0],None,[180],[0,180])
    roi_hist_s = cv2.calcHist(roi_hsv[:,:,1],[0],None,[256],[0,256])
    roi_hist_v = cv2.calcHist(roi_hsv[:,:,2],[0],None,[256],[0,256])
    max_h = np.argmax(roi_hist_h)
    max_s = np.argmax(roi_hist_s)
    max_v = np.argmax(roi_hist_v)
    roi_hsv[:45,:45,0] = max_h
    roi_hsv[:45,:45,1] = max_s
    roi_hsv[:45,:45,2] = max_v
    if DEBUG:
      cv2.imshow("roi_hsv", imutils.resize(roi_hsv, width=400))

    diff = dict()
    diff['euc'] = dist.euclidean(template_hist.flatten(), roi_hist.flatten())
    diff['man'] = dist.cityblock(template_hist.flatten(), roi_hist.flatten())
    diff['cbs'] = dist.chebyshev(template_hist.flatten(), roi_hist.flatten())
    diff['csq'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_CHISQR)
    diff['bha'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_BHATTACHARYYA)
    
    # following algo results to be interpreted in reverse
    diff['cor'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_CORREL)
    diff['int'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_INTERSECT)
    
    print(diff)
    print("Required min confidence={}%".format(args["confidence"]))
    
    if diff['cor'] >= args["confidence"]/100:
      
      print("Current match ({:.02f}%) satisfying sufficient confidence".format(100*diff['cor']))
      
      name_roi_gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
      _,name_roi_thresh = cv2.threshold(name_roi_gray, Text_Threshold, 255, cv2.THRESH_BINARY)
      cost_roi_gray = cv2.cvtColor(cost_roi, cv2.COLOR_BGR2GRAY)
      _,cost_roi_thresh = cv2.threshold(cost_roi_gray, Text_Threshold, 255, cv2.THRESH_BINARY)
      if DEBUG:
        cv2.imshow("name roi thresh", imutils.resize(name_roi_thresh, width=400))
        cv2.imshow("cost roi thresh", imutils.resize(cost_roi_thresh, width=400))
      
      # org_img = name_roi.copy()
      org_img = cost_roi.copy()
      
      (origH, origW) = org_img.shape[:2]
    
      # set the new width and height and then determine the ratio in change
      # for both the width and height
      (newW, newH) = (args["width"], args["height"])
      rW = origW / float(newW)
      rH = origH / float(newH)
      
      # resize the image and grab the new image dimensions
      resized_img = cv2.resize(org_img, (newW, newH))
      (H, W) = resized_img.shape[:2]
      
      # construct a blob from the image and then perform a forward pass of the model to obtain the two output layer sets
      # convert image to 3 channels if grayscale
      resized_img = cv2.merge([resized_img,resized_img,resized_img]) if len(resized_img.shape) == 2 else resized_img
      
      blob = cv2.dnn.blobFromImage(resized_img, 1.0, (W, H),
          (123.68, 116.78, 103.94), swapRB=True, crop=False)
      net.setInput(blob)
      
      print("Detecting and Decoding Text with min confidence={:.2f}%...".format(100*args["min_confidence"]))
      (scores, geometry) = net.forward(layerNames)
      
      # decode the predictions, then  apply non-maxima suppression to suppress weak, overlapping bounding boxes
      (rects, confidences) = decode_predictions(scores, geometry)
      rect2prob = dict(zip(rects, confidences))
      boxes = non_max_suppression(np.array(rects), probs=confidences)
      
      # initialize the list of results
      results = []
      text = None
      
      if len(boxes) == 0:
        print("No text regions detected with provided confidence")
      else:
        print("{} text regions detected with provided confidence".format(len(boxes)))
      
      # loop over the bounding boxes
      for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective ratios
        prob = rect2prob[tuple((startX, startY, endX, endY))]
        if DEBUG:
          print("Text box confidence={:.02f}".format(prob))
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially apply a bit of padding surrounding the bounding box -- here we are computing the deltas in both the x and y directions
        dX = int((endX - startX) * args["padding"])
        dY = int((endY - startY) * args["padding"])
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        # extract the actual padded ROI
        roi = org_img[startY:endY, startX:endX]
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, 
        # (2) an OEM flag of 4, indicating that the we wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        # add the bounding box coordinates and OCR'd text to the list of results
        results.append(((startX, startY, endX, endY), text))
      # sort the results bounding box coordinates from top to bottom
      results = sorted(results, key=lambda r:r[0][1])
      # loop over the results
      for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        print("{}\n".format(text))
        # strip out non-ASCII text so we can draw the text on the image using OpenCV, then draw the text and a bounding box surrounding the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = org_img.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # show the output image
        cv2.imshow("Text Detection", output)
        
        # decode the cost value by discarding all non numerals
        text = "".join([c if ord(c) in range(48,58) else "" for c in text]).strip()
        print("Decoded price value:",text)
        # cv2.waitKey(0)
        
      # cv2.rectangle(clone,(x1,y1),(x1+tw,y1+th),(0,255,0),3)
      cv2.rectangle(clone,(x1,y1),(x2,y2),(0,255,0),3)
      cv2.putText(clone, str(cntidx+1)+" --> MATCH", (x1,y1 - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
      if text != None:
        cv2.putText(clone, "COST={}".format(text), (x1+200,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
      print("Current match ({:.02f}%) not satisfying sufficient confidence".format(100*diff['cor']))
      cv2.rectangle(clone,(x1,y1),(x2,y2),(0,0,255),3)
      cv2.putText(clone, str(cntidx+1)+" ----> NO MATCH", (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
      
    cv2.imshow("match result", imutils.resize(clone, width=1200))
      
    key = cv2.waitKey(0)
    if key == 27:
      print("exiting on escape key press")
      break
    cv2.destroyAllWindows()
    plt.close("all")

def match_image_template(full_image,image,template,net,layerNames):
  ih,iw,ic = image.shape
  th,tw,tc = template.shape
  num_bins = 16
  
  #process template image
  template_blur = cv2.GaussianBlur(template,(199,199),0)
  template_hsv = cv2.cvtColor(template_blur, cv2.COLOR_BGR2HSV)
  # template_hist = cv2.calcHist(template_hsv,[0],None,[180],[0,180])
  template_hist = cv2.calcHist(template_hsv,[0,1,2],None,[num_bins,num_bins,num_bins],[0,180,0,256,0,256])
  # normalize the histogram
  template_hist /= template_hist.sum()
  # plt.figure()
  # plt.plot(template_hist)
  # plt.title("Template Histogram")
  # replace the top left of template hist with max values of hist
  template_hist_h = cv2.calcHist(template_hsv[:,:,0],[0],None,[180],[0,180])
  template_hist_s = cv2.calcHist(template_hsv[:,:,1],[0],None,[256],[0,256])
  template_hist_v = cv2.calcHist(template_hsv[:,:,2],[0],None,[256],[0,256])
  max_h = np.argmax(template_hist_h)
  max_s = np.argmax(template_hist_s)
  max_v = np.argmax(template_hist_v)
  template_hsv[:45,:45,0] = max_h
  template_hsv[:45,:45,1] = max_s
  template_hsv[:45,:45,2] = max_v
  if DEBUG:
    cv2.imshow("template_hsv", imutils.resize(template_hsv, width=400))
  
  result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
  
  clone = image.copy()
  
  loc = np.where(result >= (args["confidence"]/100))
  
  loc2prob = dict()
  loc_rects = []
  
  if min(len(loc[0]), len(loc)) == 0:
    print("No confident matches found")
  else:
    print("Confident Matches found")
    # prepare list of bounding box rectangles for non-maxima separation
    for (y,x) in zip(*loc):
      loc2prob[(x,y,x+tw,y+th)] = result[y,x]
    
  if len(loc2prob) > 0:
    loc_rects = non_max_suppression(np.array(list(loc2prob.keys())),probs=list(loc2prob.values()))
    # loc_rects = non_max_suppression(np.array(list(loc2prob.keys())),probs=None)
  
  for (x1,y1,x2,y2) in loc_rects:
    cv2.rectangle(clone,(x1,y1),(x1+tw,y1+th),(0,255,0),3)
    curr_prob = loc2prob[tuple((x1,y1,x2,y2))]
    print("Box=({},{},{},{}), Conf={:.2f}%".format(x1,y1,x2,y2,curr_prob*100))
  
  # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
  # if DEBUG:
  #   print("Maxval,MaxLoc={:.3},({},{})".format(maxVal,maxLoc[0],maxLoc[1]))
  
  # if maxVal >= args["confidence"]/100:
    # print("Confident Match found")
    # cv2.rectangle(clone, maxLoc, (maxLoc[0] + tw, maxLoc[1] + th), (0, 255, 0), 3)
    
    # obj_roi = image[maxLoc[1]:maxLoc[1]+th, maxLoc[0]:iw//2,:]
    # name_roi = image[maxLoc[1]:maxLoc[1]+(th//2), maxLoc[0]+tw:(iw//2),:]
    # cost_roi = image[maxLoc[1]+(th//2):maxLoc[1]+th, maxLoc[0]+tw:(iw//2),:]
    
    if x1 >= iw//2:
      print("Skipping incorrect match ({},{},{},{})".format(x1,y1,x2,y2))
      continue
    
    obj_roi = full_image[y1:y1+th, x1:iw//2,:]
    name_roi = full_image[y1:y1+(th//2), x1+tw:(iw//2),:]
    cost_roi = full_image[y1+(th//2):y1+th, x1+tw:(iw//2),:]
    obj_roi_short = full_image[y1:y1+th, x1:x1+tw,:]
    if DEBUG:
      cv2.imshow("match result", imutils.resize(clone, width=800))
      cv2.imshow("match roi {:.2f}".format(curr_prob), imutils.resize(obj_roi, width=400))
      cv2.imshow("name roi", imutils.resize(name_roi, width=400))
      cv2.imshow("cost roi", imutils.resize(cost_roi, width=400))
      cv2.imshow("obj_roi_short", imutils.resize(obj_roi_short, width=400))
    
    name_roi_gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    _,name_roi_thresh = cv2.threshold(name_roi_gray, Text_Threshold, 255, cv2.THRESH_BINARY)
    cost_roi_gray = cv2.cvtColor(cost_roi, cv2.COLOR_BGR2GRAY)
    _,cost_roi_thresh = cv2.threshold(cost_roi_gray, Text_Threshold, 255, cv2.THRESH_BINARY)
    if DEBUG:
      cv2.imshow("name roi thresh", imutils.resize(name_roi_thresh, width=400))
      cv2.imshow("cost roi thresh", imutils.resize(cost_roi_thresh, width=400))
    
    # org_img = name_roi.copy()
    org_img = cost_roi.copy()
    
    (origH, origW) = org_img.shape[:2]
  
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = origW / float(newW)
    rH = origH / float(newH)
    
    # resize the image and grab the new image dimensions
    resized_img = cv2.resize(org_img, (newW, newH))
    (H, W) = resized_img.shape[:2]
    
    # construct a blob from the image and then perform a forward pass of the model to obtain the two output layer sets
    # convert image to 3 channels if grayscale
    resized_img = cv2.merge([resized_img,resized_img,resized_img]) if len(resized_img.shape) == 2 else resized_img
    
    blob = cv2.dnn.blobFromImage(resized_img, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    
    print("Detecting and Decoding Text ...")
    (scores, geometry) = net.forward(layerNames)
    
    # decode the predictions, then  apply non-maxima suppression to suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    # initialize the list of results
    results = []
    
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
      # scale the bounding box coordinates based on the respective ratios
      startX = int(startX * rW)
      startY = int(startY * rH)
      endX = int(endX * rW)
      endY = int(endY * rH)
  
      # in order to obtain a better OCR of the text we can potentially apply a bit of padding surrounding the bounding box -- here we are computing the deltas in both the x and y directions
      dX = int((endX - startX) * args["padding"])
      dY = int((endY - startY) * args["padding"])
  
      # apply padding to each side of the bounding box, respectively
      startX = max(0, startX - dX)
      startY = max(0, startY - dY)
      endX = min(origW, endX + (dX * 2))
      endY = min(origH, endY + (dY * 2))
  
      # extract the actual padded ROI
      roi = org_img[startY:endY, startX:endX]
  
      # in order to apply Tesseract v4 to OCR text we must supply
      # (1) a language, 
      # (2) an OEM flag of 4, indicating that the we wish to use the LSTM neural net model for OCR, and finally
      # (3) an OEM value, in this case, 7 which implies that we are treating the ROI as a single line of text
      config = ("-l eng --oem 1 --psm 7")
      text = pytesseract.image_to_string(roi, config=config)
  
      # add the bounding box coordinates and OCR'd text to the list of results
      results.append(((startX, startY, endX, endY), text))
    
    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])
    
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
      # display the text OCR'd by Tesseract
      print("OCR TEXT")
      print("========")
      print("{}\n".format(text))
  
      # strip out non-ASCII text so we can draw the text on the image using OpenCV, then draw the text and a bounding box surrounding the text region of the input image
      text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
      output = org_img.copy()
      cv2.rectangle(output, (startX, startY), (endX, endY),
          (0, 0, 255), 2)
      cv2.putText(output, text, (startX, startY + 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
  
      # show the output image
      cv2.imshow("Text Detection", output)
      # cv2.waitKey(0)
      
    # start histogram calculation and comparison
    roi_blur = cv2.GaussianBlur(obj_roi_short,(199,199),0)
    roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)    
    # roi_hist = cv2.calcHist(roi_hsv,[0],None,[180],[0,180])
    roi_hist = cv2.calcHist(roi_hsv,[0,1,2],None,[num_bins,num_bins,num_bins],[0,180,0,256,0,256])
    # normalize the histogram
    roi_hist /= roi_hist.sum()
    # plt.figure()
    # plt.plot(roi_hist)
    # plt.title("ROI Histogram")
    # replace the top left of roi hist with max values from hist
    roi_hist_h = cv2.calcHist(roi_hsv[:,:,0],[0],None,[180],[0,180])
    roi_hist_s = cv2.calcHist(roi_hsv[:,:,1],[0],None,[256],[0,256])
    roi_hist_v = cv2.calcHist(roi_hsv[:,:,2],[0],None,[256],[0,256])
    max_h = np.argmax(roi_hist_h)
    max_s = np.argmax(roi_hist_s)
    max_v = np.argmax(roi_hist_v)
    roi_hsv[:45,:45,0] = max_h
    roi_hsv[:45,:45,1] = max_s
    roi_hsv[:45,:45,2] = max_v
    if DEBUG:
      cv2.imshow("roi_hsv", imutils.resize(roi_hsv, width=400))

    diff = dict()
    diff['euc'] = dist.euclidean(template_hist.flatten(), roi_hist.flatten())
    diff['man'] = dist.cityblock(template_hist.flatten(), roi_hist.flatten())
    diff['cbs'] = dist.chebyshev(template_hist.flatten(), roi_hist.flatten())
    diff['csq'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_CHISQR)
    diff['bha'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_BHATTACHARYYA)
    
    # following algo results to be interpreted in reverse
    diff['cor'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_CORREL)
    diff['int'] = cv2.compareHist(template_hist.flatten(), roi_hist.flatten(), cv2.HISTCMP_INTERSECT)
    
    print(diff)
    
    key = cv2.waitKey(0)
    if key == 27:
      print("exiting on escape key press")
      break
    cv2.destroyAllWindows()
    plt.close("all")
  
  # else:
  #   print("No confident match found")

# define the two output layer names for the EAST detector model that we are interested -- the first is the output probabilities and the second can be used to derive the bounding box coordinates of text
txt_layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
txt_net = cv2.dnn.readNet(args["detector"])

in_image = cv2.imread(args["image"])
in_template = cv2.imread(args["template"])

b = in_image.copy()
# b = cv2.GaussianBlur(image, (3,3), 0);
# b = cv2.bilateralFilter(image, 31,51,51)
b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
if DEBUG:
  cv2.imshow("Image processed",imutils.resize(b,width=800))
  cv2.imshow("Image gray",imutils.resize(b,width=800))
# c = imutils.auto_canny(b);
# c = cv2.Canny(b,10,200);
c = cv2.Canny(b,30,150);
# c = cv2.Canny(b,240,250);
cntlst = cv2.findContours(c.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntlst = imutils.grab_contours(cntlst)
d = np.zeros(b.shape, dtype=np.uint8)
cv2.drawContours(d, cntlst, -1, (255, 255, 255), 2)
if DEBUG:
  cv2.imshow("Contours",imutils.resize(d,width=800))

cntlst_approx = []
e = np.zeros(b.shape, dtype=np.uint8)
for cnt in cntlst:
  peri = cv2.arcLength(cnt,True)
  tmp = cv2.approxPolyDP(cnt,0.01*peri,True)
  if len(tmp) == 4:
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspectRatio = w / float(h)
    area = cv2.contourArea(cnt)
    if aspectRatio >= 0.8 and aspectRatio <=1.2 and area >= 10000:
      cntlst_approx.append(cnt)
      # print("area=",cv2.contourArea(cnt))
      cv2.drawContours(e, [cnt], -1, (255, 255, 255), -1)
    
f = cv2.bitwise_and(in_image,in_image,mask=e)
if DEBUG:
  cv2.imshow("Mask",imutils.resize(e,width=800))
  cv2.imshow("Extracted ROIs",imutils.resize(f,width=800))

# match_image_template(in_image,f,in_template,txt_net,txt_layerNames)
match_image_hsv(in_image,e,in_template,txt_net,txt_layerNames)

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close("all")
