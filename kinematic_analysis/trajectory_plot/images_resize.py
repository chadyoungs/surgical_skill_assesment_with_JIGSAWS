# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:55:25 2020
@author: xiaoxiaoyang
"""

import os
from PIL import Image
 
# 640*480
IMG_HEIGHT = 1200
IMG_WIDTH = 3600

TARGET_HEIGHT = 120
TARGET_WIDTH = 360

out = Image.open('image_stitch.jpg').resize((TARGET_WIDTH, TARGET_HEIGHT)).save('image_stitch.jpg')


