# Usage
# Resnet18.py --input bird1.jpg

from torchvision import models
from PIL import Image
import cv2
import torch
from torchsummary import summary

from torchvision import transforms

import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path of the image to be classified")
args = vars(ap.parse_args())
image_path = args['input']

topN = 5

transform = transforms.Compose([            #[1]
# transforms.Resize(256),                    #[2]
# transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

dir(models)

img = Image.open(image_path)

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# First, load the model
resnet = models.resnet18(pretrained=True)
# summary(resnet, (3, 224,224))

# Second, put the network in eval mode
resnet.eval()

# Third, carry out model inference
preds = resnet(batch_t)
preds = torch.softmax(preds, dim=1)
pred, class_idx = torch.max(preds, dim=1)

p = preds.sort(dim = 1, descending=True)

print('Top {} predictions:'.format(topN))
for i in range(topN):
  pred = p[0][0,i]
  class_idx = p[1][0,i]
  print('Predicted Class:[{}], Prob:[{:0.03}%]'.format(labels[class_idx],100*pred.item()))

