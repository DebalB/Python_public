# Usage
# ReceptiveFieldObjectDetector.py --input bird5.jpg
# ReceptiveFieldObjectDetector.py --input camel.jpg

import numpy as np
import cv2
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import transforms
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18, showTopNPreds
import argparse
from imutils import object_detection
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path of the image to be classified")
args = vars(ap.parse_args())
image_path = args['input']

Rect = namedtuple('Rect', 'x1 y1 x2 y2')

saveModel = False
saveDetection = False
displayWidth = 1024
# useGpu = False
useGpu = True
predThresh = 0.20
useMaxActivations = False

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
        # obj_img = (obj_detected*255.0).astype('uint8')
        # cv2.imshow("obj_detected",imutils.resize(obj_detected,width=displayWidth))
        # cv2.waitKey(0)
        
        rects = find_rect(obj_detected)
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
      # print('moving to GPU {model, input, scoremap}')
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


def find_rect(activations):
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


def visualize_activations(image, activations, show_bounding_rect=False):
    activations = normalize(activations)

    activations_multichannel = np.stack([activations, activations, activations], axis=2)
    masked_image = (image * activations_multichannel).astype(np.uint8)

    if show_bounding_rect:
        rect = find_rect(activations)
        if rect is not None:
          cv2.rectangle(masked_image, (rect.x1, rect.y1), (rect.x2, rect.y2), color=(0, 0, 255), thickness=2)
        else:
          print('Bounding rectangle not detected')

    return masked_image


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

        # Find the class with the maximum score in the n x m output map
        # pred_max, class_idx = torch.max(preds, dim=1)

        # row_max, row_idx = torch.max(pred_max, dim=1)
        # col_max, col_idx = torch.max(row_max, dim=1)
        # predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]
        # pred_prob = pred_max[0, row_idx[0, col_idx], col_idx]

        # Print the top predicted class
        # print('Predicted Class:[{}], Value:[{}], Probability:[{:0.03}%]'.format(labels[predicted_class], predicted_class.item(),100*pred_prob.item()))
        
        # showTopNPreds(pred_max,class_idx,labels,topN=5)

        # Find the n x m score map for the predicted class
        # score_map = preds[0, predicted_class, :, :].cpu()

    # Compute the receptive filed for the inference result for max activated pixel
    # receptive_field_map = backprop_receptive_field(image, scoremap=score_map, predicted_class=predicted_class)
    
    # Compute the receptive filed for the inference result for the net prediction
    # receptive_field_net_pred_map = backprop_receptive_field(image, scoremap=score_map, predicted_class=predicted_class, use_max_activation=False)

    # Resize score map to the original image size
    # score_map = score_map.numpy()[0]
    # score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))

    # Display the images
    # cv2.imshow("Original Image", imutils.resize(original_image,width=displayWidth))
    
    # cv2.imshow("Score map: activations and bbox", imutils.resize(visualize_activations(original_image, score_map,show_bounding_rect=True),width=displayWidth))
    
    # cv2.imshow("receptive_field_max_activation", imutils.resize(visualize_activations(original_image, receptive_field_map, show_bounding_rect=True),width=displayWidth))
    
    # cv2.imshow("receptive_field_net_prediction", imutils.resize(visualize_activations(original_image, receptive_field_net_pred_map, show_bounding_rect=True),width=displayWidth))
    
    # detect objects and visualize detections
    detect_objects(original_image,image,preds,labels)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def main():
    # Read the image
    image = cv2.imread(image_path)

    run_resnet_inference(image)

if __name__ == "__main__":
    main()
