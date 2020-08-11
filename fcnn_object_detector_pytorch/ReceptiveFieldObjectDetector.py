# Usage
# ReceptiveFieldObjectDetector.py --input res/bird5.jpg
# ReceptiveFieldObjectDetector.py --input res/camel.jpg

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18
import argparse
from imutils import object_detection
import imutils
from utility_functions import find_rects, normalize

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path of the image to be classified")
args = vars(ap.parse_args())
image_path = args['input']

useGpu = True
saveModel = False
saveDetection = False
useMaxActivations = True
displayWidth = 1024
predThresh = 0.20

def detect_objects(original_image,image_tensor,preds,categories):
  
  labels = {}
  
  # preds = preds.squeeze()
  preds_max,labels_max = torch.max(preds,dim= 1)
  # print("preds_max shape:",preds_max.shape)
  
  numrows = preds_max[0].shape[0]
  numcols = preds_max[0].shape[1]
  cnt = 0
  
  for rowidx in range(numrows):
    for colidx in range(numcols):
      pmax = preds_max[0,rowidx,colidx]
      class_val = labels_max[0,rowidx,colidx]
      
      if pmax > predThresh:
        cnt+=1
        print('[{}]Class:[{}], Prob:[{:.03}%], Value:[{}]'.format(cnt,categories[class_val].split(',')[0],100*pmax,class_val))
      
        class_val = torch.tensor([class_val])
        score_map = preds[0, class_val, :, :].cpu()
        max_idx = (rowidx,colidx)
        
        obj_detected = backprop_receptive_field(image_tensor,scoremap=score_map, predicted_class=class_val,use_max_activation=useMaxActivations,max_loc=max_idx)
        
        obj_detected = normalize(obj_detected)
        
        # visualize interim detection
        obj_img = (obj_detected*255.0).astype('uint8')
        cv2.imshow("obj_detected",imutils.resize(obj_img,width=displayWidth))
        cv2.waitKey(0)
        
        rects = find_rects(obj_detected)
        for rect in rects:
          box = (rect.x1, rect.y1, rect.x2, rect.y2)
          L = labels.get(class_val, [])
          L.append((box, pmax))
          labels[class_val] = L
  
  clone = original_image.copy()
  
  for label in labels.keys():
    color = np.random.randint(0,100,size=(3))
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = object_detection.non_max_suppression(boxes, proba)
    
    for (startX, startY, endX, endY) in boxes:
      # draw the bounding box and label on the image
      cv2.rectangle(clone, (startX, startY), (endX, endY), [int(color[0]),int(color[1]),int(color[2])], 2)
      y = startY - 10 if startY - 10 > 10 else startY + 10
      text = categories[label].split(',')[0]
      cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [int(color[0]),int(color[1]),int(color[2])], 2)

  # show the output after apply non-maxima suppression
  cv2.imshow("Objects Detected", imutils.resize(clone,width=displayWidth))
  
  if saveDetection == True:
    cv2.imwrite('detection_result.png',clone)
    # cv2.waitKey(0)

def backprop_receptive_field(image, predicted_class, scoremap, use_max_activation=True,max_loc=None):
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


def run_resnet_inference(original_image):
    # Read ImageNet class id to name mapping
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    # Convert original image to RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Transform input image
    # 1. Convert to Tensor
    # 2. Subtract mean
    # 3. Divide by standard deviation

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Subtract mean
            std=[0.229, 0.224, 0.225]  # Divide by standard deviation
        )])

    image = transform(image)
    image = image.unsqueeze(0)

    # Load modified resnet18 model with pretrained ImageNet weights
    model = FullyConvolutionalResnet18(pretrained=True).eval()
    
    if saveModel == True:
      torch.save(model,'modelfile.pt')
    
    if useGpu == True and torch.cuda.is_available() == True:
      # print('moving to GPU {model,image}')
      model = model.cuda()
      image = image.cuda()

    with torch.no_grad():
        # Perform the inference.
        # Instead of a 1x1000 vector, we will get a
        # 1x1000xnxm output ( i.e. a probabibility map
        # of size n x m for each 1000 class,
        # where n and m depend on the size of the image.)
        preds = model(image)
        preds = torch.softmax(preds, dim=1)
    
    # detect objects and visualize detections
    detect_objects(original_image,image,preds,labels)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def main():
    # Read the image
    if not os.path.exists(image_path):
      print('Err: Input Image file not found:',image_path)
      return
    
    image = cv2.imread(image_path)

    run_resnet_inference(image)

if __name__ == "__main__":
    main()
