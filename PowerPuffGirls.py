
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:06:09 2017

@author: Wenjing Shi
"""

#import numpy as np

from FastSmallVideoSim import FastSmallVideoSim
import cv2

import pickle
#from matplotlib import pyplot as plt


back_img = cv2.imread('City.jpg')
print(back_img.shape)

#obj_img = cv2.imread('Monkey_1.png')

#obj_img_1 = cv2.imread('b.jpg')
#obj_img_2 = cv2.imread('r.jpg')
#obj_img_3 = cv2.imread('g.jpg')

obj_img_1 = pickle.load(open("obj_B.pkl","rb"))


obj_img_2 = pickle.load(open("obj_R.pkl","rb"))
obj_img_3 = pickle.load(open("obj_G.pkl","rb"))
obj_img = pickle.load(open("obj_MONKEY.pkl","rb"))

print(obj_img_1.shape)
print(obj_img_2.shape)
print(obj_img_3.shape)
print(obj_img.shape)

obj_roi_1 = pickle.load(open("mask_B.pkl","rb"))
obj_roi_2 = pickle.load(open("mask_R.pkl","rb"))
obj_roi_3 = pickle.load(open("mask_G.pkl","rb"))
obj_roi = pickle.load(open("mask_MONKEY.pkl","rb"))

(back_rows, back_cols) = back_img.shape[:2]
(back_row_center, back_col_center) = (back_rows/2, back_cols/2)

(row_center_1, col_center_1) = (obj_img_1.shape[0]/2, obj_img_1.shape[1]/2)

(row_center_2, col_center_2) = (obj_img_2.shape[0]/2, obj_img_2.shape[1]/2)
(row_center_3, col_center_3) = (obj_img_3.shape[0]/2, obj_img_3.shape[1]/2)

(row_center, col_center) = (obj_img.shape[0]/2, obj_img.shape[1]/2)
print('bubble',row_center_1, col_center_1)
print('bloom',row_center_2, col_center_2)
print('buttercup', row_center_3, col_center_3)
print(row_center, col_center)
        
fsvs = FastSmallVideoSim('PowerPuff.avi', back_rows, back_cols)
fsvs.add_back('City', back_img)
fsvs.add_obj('Monkey', obj_img, obj_roi) 
fsvs.add_obj('Bubbles', obj_img_1, obj_roi_1) 
fsvs.add_obj('Blossom', obj_img_2, obj_roi_2)
fsvs.add_obj('Buttercup', obj_img_3, obj_roi_3)



""" Create the first frame"""
e = cv2.getTickCount()
frame_0 = fsvs.init_video('City', 0, 0)
e0 = cv2.getTickCount()
fsvs.size()

""" Create the second frame"""
fsvs.new_frame(0, 0)
fsvs.place_obj('Bubbles', 350, 500)
fsvs.place_obj('Blossom', 350, 250)
fsvs.place_obj('Buttercup', 350, 0)


e1 = cv2.getTickCount()


""" Create the third frame"""
Scale_Val_1 = 1
for i in range (1, 70):  #70
    Scale_Val_1 = Scale_Val_1 - 0.01
    fsvs.new_frame(0, 0)  

   
    (new_row_1, new_col_1) = fsvs.move_obj('Bubbles', -5, -6)
    (new_row_2, new_col_2) = fsvs.move_obj('Blossom', -5, 0)
    (new_row_3, new_col_3) = fsvs.move_obj('Buttercup', -5, 6)

    fsvs.place_obj('Buttercup', new_row_3, new_col_3)
    fsvs.place_obj('Bubbles', new_row_1, new_col_1)
    fsvs.place_obj('Blossom', new_row_2, new_col_2)
   

e2 = cv2.getTickCount()

Scale_Val = 1
""" Create the forth frame"""

for i in range (1, 360, 2):
    
    Scale_Val = Scale_Val - 0.005
    fsvs.new_frame(0, 0)

    
    (new_scl_name_1,row_delta_1, col_delta_1) = fsvs.rot_obj('Bubbles', row_center_1, col_center_1, i)
    (new_scl_name_2,row_delta_2, col_delta_2) = fsvs.rot_obj('Blossom', row_center_2, col_center_2, i)
    (new_scl_name_3,row_delta_3, col_delta_3) = fsvs.rot_obj('Buttercup', row_center_3, col_center_3, i)

    

    (new_scl_name_1,row_delta_1_1, col_delta_1_1)  = fsvs.scale_obj(new_scl_name_1, row_center_1, col_center_1, Scale_Val)
    (new_scl_name_2,row_delta_2_1, col_delta_2_1)  = fsvs.scale_obj(new_scl_name_2, row_center_2, col_center_2, Scale_Val)
    (new_scl_name_3,row_delta_3_1, col_delta_3_1)  = fsvs.scale_obj(new_scl_name_3, row_center_3, col_center_3, Scale_Val)
    
    """ for only rotation or scaling """
    #fsvs.place_obj(new_scl_name_1, int(new_row_1-row_delta_1), int(new_col_1-col_delta_1))
    #fsvs.place_obj(new_scl_name_2, int(new_row_2-row_delta_2), int(new_col_2-col_delta_2)) 
    #fsvs.place_obj(new_scl_name_3, int(new_row_3-row_delta_3), int(new_col_3-col_delta_3))
    
    """ for rotation and scaling """
    fsvs.place_obj(new_scl_name_1, int(new_row_1-row_delta_1-row_delta_1_1), int(new_col_1-col_delta_1-col_delta_1_1))
    fsvs.place_obj(new_scl_name_2, int(new_row_2-row_delta_2-row_delta_2_1), int(new_col_2-col_delta_2-col_delta_2_1)) 
    fsvs.place_obj(new_scl_name_3, int(new_row_3-row_delta_3-row_delta_3_1), int(new_col_3-col_delta_3-col_delta_3_1))

     



e3 = cv2.getTickCount()

""" Create the fifth frame"""

for i in range (1, 40):
    
    fsvs.new_frame(0, 0)
    (new_scl_monkey_name_1, row_delta, col_delta)  = fsvs.scale_obj('Monkey', row_center, col_center, 0.2)
    fsvs.place_obj(new_scl_monkey_name_1, 100, 300)
    
    
e4 = cv2.getTickCount()

for i in range (1, 40):
    fsvs.new_frame(0, 0)
    (new_scl_monkey_name_2, row_delta, col_delta)  = fsvs.scale_obj('Monkey', row_center, col_center, 0.3)

    (new_scl_rot_monkey_name_2, row_delta, col_delta)= fsvs.rot_obj(new_scl_monkey_name_2, fsvs.dic_of_obj_imgs[new_scl_monkey_name_2][1].shape[0]/2, fsvs.dic_of_obj_imgs[new_scl_monkey_name_2][1].shape[1]/2, 90)
    fsvs.place_obj(new_scl_rot_monkey_name_2, 300, 450)
    
    
    
e5 = cv2.getTickCount()

for i in range (1, 40):
    fsvs.new_frame(0, 0)
 
    
    (new_rot_monkey_1, row_delta, col_delta) = fsvs.rot_obj('Monkey', row_center, col_center, 315)
    
    (new_rot_scl_monkey_1, row_delta, col_delta)  = fsvs.scale_obj(new_rot_monkey_1, row_center, col_center, 0.5)

    fsvs.place_obj(new_rot_scl_monkey_1, 200, -150)
    


e6 = cv2.getTickCount()

for i in range (1, 40):
    fsvs.new_frame(0, 0)
 
    (new_scl_monkey_name_3, row_delta, col_delta) = fsvs.scale_obj('Monkey', row_center, col_center, 0.8)

    fsvs.place_obj(new_scl_monkey_name_3, 50, -30)
    
    
e7 = cv2.getTickCount()


fsvs.save_video()
fsvs.play_video()
fsvs.out.release()
cv2.destroyAllWindows()


""" Time performance"""
time_0 = (e0 - e)/ cv2.getTickFrequency()
time_1 = (e1 - e0)/ cv2.getTickFrequency()
time_2 = (e2 - e1)/ cv2.getTickFrequency()/70
time_3 = (e3 - e2)/ cv2.getTickFrequency()/180
time_4 = (e4 - e3)/ cv2.getTickFrequency()/40
time_5 = (e5 - e4)/ cv2.getTickFrequency()/40
time_6 = (e6 - e5)/ cv2.getTickFrequency()/40
time_7 = (e7 - e6)/ cv2.getTickFrequency()/40
total_time =(e7 - e)/ cv2.getTickFrequency() 

print("time_0 = ", time_0, "   freq = ", 1/time_0)
print("time_1 = ", time_1, "   freq = ", 1/time_1)
print("time_2 = ", time_2, "   freq = ", 1/time_2)
print("time_3 = ", time_3, "   freq = ", 1/time_3)
print("time_4 = ", time_4, "   freq = ", 1/time_4)
print("time_5 = ", time_5, "   freq = ", 1/time_5)
print("time_6 = ", time_6, "   freq = ", 1/time_6)
print("time_7 = ", time_7, "   freq = ", 1/time_7)
print("total_time = ", total_time)