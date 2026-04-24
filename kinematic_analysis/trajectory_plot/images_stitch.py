# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:55:25 2020
@author: xiaoxiaoyang
"""

import os
from PIL import Image
 
# 640*480
IMG_HEIGHT = 1200
IMG_WIDTH = 1800
TARGET_WIDTH = 2 * IMG_WIDTH
#TARGET_WIDTH += 20

task_list = ["Suturing", "Knot_Tying", "Needle_Passing"]

# 0 for suturing
TASK_SYMBOL = 0
 
image_list = ["3D trajectories of left hand of expert and novice surgeon.png", 
              "3D trajectories of right hand of expert and novice surgeon.png"]

target = Image.new('RGB',(TARGET_WIDTH, IMG_HEIGHT))
left = 0
right = IMG_WIDTH
for count, image in enumerate(image_list):
    target.paste(Image.open(image), (left, 0, right, IMG_HEIGHT))
    
    left += IMG_WIDTH
    right += IMG_WIDTH
    target.paste((255, 255, 255), (left, 0, right, IMG_HEIGHT))
    if count == 1:
        break
    left += 10
    right += 10
    
quantity_value = 100
target.save('image_stitch.jpg', quantity = quantity_value)

