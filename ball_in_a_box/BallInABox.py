# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:33:56 2020

@author: DEBAL
"""

from vpython import *

showTrail = False
autoScale = True
showArrow = False

displayRate = 100
rightWallX = 11.5
leftWallX = -rightWallX
topWallY = 7
bottomWallY = -topWallY
backWallZ = -10.90
frontWallZ = 4

ball = sphere(pos=vector(0,0,0), radius=0.5, color=color.cyan, make_trail=showTrail)
wallR = box(pos=vector(rightWallX,0,-4), size=vector(0.2,13.8,14), color=color.green) 
wallL = box(pos=vector(leftWallX,0,-4), size=vector(0.2,13.8,14), color=color.green)
wallT = box(pos=vector(0,topWallY,-4), size=vector(23.25,0.2,14), color=color.blue)
wallB = box(pos=vector(0,bottomWallY,-4), size=vector(23.25,0.2,14), color=color.blue)
wallBack = box(pos=vector(0,0,backWallZ), size=vector(23.25,13.75,0.2), color=color.red)

ball.velocity = vector(25,15,5) 
deltat = 0.005
t = 0 
vscale = 0.1

if showArrow == True:
  varr = arrow(pos=ball.pos, axis=vscale*ball.velocity, color=color.yellow) 

scene.autoscale = autoScale

# while t < 10:
while True:
  rate(displayRate)
  ball.pos = ball.pos + ball.velocity*deltat
  if (ball.pos.x+ball.radius > wallR.pos.x) or (ball.pos.x-ball.radius < wallL.pos.x):
    ball.velocity.x = -ball.velocity.x
    
  if (ball.pos.y+ball.radius > wallT.pos.y) or (ball.pos.y-ball.radius < wallB.pos.y):
    ball.velocity.y = -ball.velocity.y
    
  if (ball.pos.z+ball.radius > frontWallZ) or (ball.pos.z-ball.radius < wallBack.pos.z):
    ball.velocity.z = -ball.velocity.z
  
  if showArrow == True:
    varr.pos = ball.pos
    varr.axis = vscale*ball.velocity
  
  t+=deltat

print('finished')