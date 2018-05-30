# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:40:28 2017

@author: Wenjing Shi
"""

import cv2
import AOLME
import pickle
from matplotlib import pyplot as plt
import numpy as np


obj_img = cv2.imread('b.jpg')
obj_roi = AOLME.ROI(obj_img)




contours = cv2.findContours(obj_roi.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
print('cnt = ', cnt)
(x,y,w,h) = cv2.boundingRect(cnt)

obj_data = np.copy(obj_img[y:(y+h), x:(x+w), :])
mask_data = np.copy(obj_roi[y:(y+h), x:(x+w)])

pickle.dump(obj_data, open("obj_B_0.pkl","wb"), protocol = 2)
pickle.dump(mask_data , open("mask_B_0.pkl","wb"), protocol = 2)


#mask1 = pickle.load(open("mask_TEST.pkl","rb"))

plt.imshow(obj_data)
plt.show()

plt.imshow(mask_data)
plt.show()




