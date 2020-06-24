# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:04:37 2020

@author: DEBAL
"""
# Usage:
# monster_game.py --images "./monsters/"

import cv2
import argparse
import imutils
from imutils import paths
import random
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the folder containing monster images")
args = vars(ap.parse_args())

# (screen_width, screen_height) = (800, 600)
(screen_width, screen_height) = (1360, 768)
border_size = 100
min_item_width = 200
rand_scales = np.linspace(0.3, 0.6, 100)
min_weapon_area = 5000
# create_interval = 100
create_interval = 25
game_lose_cnt = 50
game_win_cnt = 5000

monster_list = []
monster_images = []

for imagePath in sorted(list(paths.list_images(args["images"]))):
  image = cv2.imread(imagePath)
  if image.shape[1] < min_item_width:
    image = imutils.resize(image, width=min_item_width)
  # cv2.imshow("monster", image)
  monster_images.append(image)
  cv2.waitKey(1)

cv2.destroyAllWindows()

def add_monster_to_list(image, width, centx, centy):
  monster_list.append([imutils.resize(image, width=width),(centx, centy)])

def draw_monsters_in_list(frame):
  for item,(centx,centy) in monster_list:
    # cv2.circle(frame,(centx,centy),item.shape[1],(255,0,0),-1)
    leftx = centx - (item.shape[1]//2)
    lefty = centy - (item.shape[0]//2)
    rightx = leftx+item.shape[1]
    righty = lefty+item.shape[0]
    # print(frame.shape, lefty, righty, leftx, rightx)
    frame[lefty:righty, leftx:rightx] = item
  # print("---")
  return frame

def find_monsters_to_kill(frame, box):
  item_del_list = []
  for item,(centx,centy) in monster_list:
    canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
    top_leftx = centx - (item.shape[1]//2)
    top_lefty = centy - (item.shape[0]//2)
    # draw monster
    cv2.rectangle(canvas, (top_leftx, top_lefty), (top_leftx+item.shape[1], top_lefty+item.shape[0]), (255,255,255), -1)
    # draw weapon
    cv2.drawContours(canvas,[box],0,(255,255,255),-2)
    # cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    # find contours in canvas
    cnts = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if (len(cnts) == 1):
      #item and weapons are in contact
      print("Found item to delete at:({})".format((centx,centy)))
      item_del_list.append([item, (centx,centy)])
  
  cv2.destroyWindow("canvas")
  return item_del_list

def remove_monsters_from_list(item_list):
  for item in item_list:
    print("Total monsters in list before:{}".format(len(monster_list)))
    print("Removing item at ({})".format(item[1]))
    try:
      monster_list.remove(item)
    except Exception as ex:
      print("!!! Got an exception while deletion from list:", ex)
      
    print("Total monsters in list after:{}".format(len(monster_list)))
    return

def get_weapon_mask():
  l_h = 16
  l_s = 83
  l_v = 92
  u_h = 58
  u_s = 255
  u_v = 255
  maskLower = np.array([l_h, l_s, l_v])
  maskUpper = np.array([u_h, u_s, u_v])
  return (maskLower, maskUpper)

def find_weapon_coordinates(frame):
  box = []
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  (maskLower, maskUpper) = get_weapon_mask()
  mask_img = cv2.inRange(hsv, maskLower, maskUpper)
  
  cnts = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts_sorted = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
  
  # bounding_rect = cv2.boundingRect(cnts_sorted[0])
  # (x, y, w, h) = bounding_rect
  # cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
  
  # compute the rotated bounding box of the region
  if len(cnts_sorted) > 0:
    if (cv2.contourArea(cnts_sorted[0]) > min_weapon_area):
      rect = cv2.minAreaRect(cnts_sorted[0])
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      cv2.drawContours(frame,[box],0,(0,0,255),2)
  
  return (frame, box)
  
def reset_game():
  monster_list = []
  create_interval = 25
  
# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)

cnt = 0
monsters_added = 0
monsters_killed = 0
refresh_flag = True
# loop over the frames of the video
while True:
  cnt+=1
  # grab the current frame
  (grabbed, frame) = camera.read()
  
  # if the frame could not be grabbed, then we have reached the end of the video
  if not grabbed:
    break
  
  # resize the frame, convert the frame to grayscale, and detect faces in the frame
  # frame = imutils.resize(frame, width=1000)
  frame = cv2.resize(frame, (screen_width,screen_height))
  frame = cv2.flip(frame,1)
  
  if refresh_flag == True and monsters_killed != 0 and monsters_killed % 5 == 0 and create_interval > 2:
    print("reducing monster creation time")
    create_interval -= 1
    refresh_flag = False
  
  if (cnt % create_interval == 0):
    # select a random width
    rand_scale = random.choice(rand_scales)
    monster_image = random.choice(monster_images)
    new_width = int(monster_image.shape[1] * rand_scale)
    # select a centroid pixel
    # cent_x = random.randrange(border_size, screen_width-border_size, 100)
    # cent_y = random.randrange(border_size, screen_height-border_size, 100)
    cent_x = int(np.random.uniform(low=border_size, high=screen_width-border_size))
    cent_y = int(np.random.uniform(low=border_size, high=screen_height-border_size))
    
    add_monster_to_list(monster_image,new_width,cent_x,cent_y)
    monsters_added += 1
  
  frame, box = find_weapon_coordinates(frame)
  
  frame = draw_monsters_in_list(frame)
  
  if (len(box) > 0):
    # print("Found a weapon {}".format(cnt))
    item_del_list = find_monsters_to_kill(frame, box)
    if len(item_del_list) > 0:
      print("Found {} monsters to kill {}".format(len(item_del_list),cnt))
      remove_monsters_from_list(item_del_list)
      monsters_killed += len(item_del_list)
      refresh_flag = True
  
  if monsters_added < monsters_killed:
    print("!!! Unexpected situation, reseting counters")
    monsters_added = monsters_killed
    monster_list = []

  text = "Monsters added:{}, Monsters killed:{}".format(monsters_added, monsters_killed)
  cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 2)

  if monsters_added - monsters_killed >= game_lose_cnt:
    cv2.putText(frame,"GAME OVER !! YOU LOSE",(screen_width//6,screen_height//2),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0, 0, 255), 5)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
  elif monsters_killed >= game_win_cnt:
    cv2.putText(frame,"GAME OVER !! YOU WIN",(screen_width//6,screen_height//2),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0, 255, 0), 5)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
  else:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
  if key == ord('q') or key == 27:
    break
  elif key == ord('r'):
    reset_game()
    cnt = 0
    refresh_flag = False
    monsters_killed = 0
    monsters_added = 0
    
camera.release()
cv2.destroyAllWindows()
