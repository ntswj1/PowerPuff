# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:51:20 2017

@author: Wenjing Shi
"""
import numpy as np
import cv2
import math
import pylab as pl
from roipoly import roipoly
import winsound
import time




def ROI(img):
    
    pl.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rp = roipoly(roicolor='r') 
    roi_mask = rp.getMask(img)
       
    return roi_mask


class PowerPuff:
    """Fast digital video simulator for small videos that fit in memory.
       Usage example: ? Provide usage example here.
    """ 
    # Initialize the dictionary of background and object images.
    dic_of_back_imgs = {}  
    dic_of_obj_imgs  = {}
    dic_of_obj_rot  = {}
    
    frame_ps = 35
    #
    
  
    #####################################
    def __init__(self, video_name, num_of_rows, num_of_cols, speed):
        """Constructor that stores video name and video size."""
        self.video_name = video_name
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.video      = []
        self.obj_locs   = {}
        self.out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), self.frame_ps, (self.num_of_cols, self.num_of_rows))
        self.px_frame = speed/self.frame_ps  # px/per_frame
        #print('(self.num_of_cols, self.num_of_rows) = ', self.num_of_cols, self.num_of_rows)
        print("(Initializing {})".format(self.video_name))
        
    def add_back(self, back_name, back_img):
        """Adds a background image for sharing."""
        self.dic_of_back_imgs[back_name] = back_img
         
        
    def add_obj(self, obj_name, obj_img, obj_roi):
        """Adds an object image for sharing."""
        obj = (obj_img, obj_roi)
        self.dic_of_obj_imgs[obj_name] = obj 
        
    def add_obj_rot(self, obj_name, direction, rot_angle):
        """Adds an object image for sharing."""
        obj = (direction, rot_angle)
        self.dic_of_obj_rot[obj_name] = obj 
        
    #############################################    
    def init_video(self, back_name, start_row, start_col):
        
        #Initialize the video to the background image given by back_name using the corresponding: background_image[start_row:start_row+num_of_rows-1, start_col:start_col+num_of_cols-1]."
        # Create the first video frame using:
        # 1. Store the name of the background image.
        # 2. Extract the video frame from the background
        #    at (start_row, start_col).
        # 3. Use self.video.append() to initialize the video frame.
        backgournd_frame = self.dic_of_back_imgs[back_name][start_row:start_row + self.num_of_rows, start_col:start_col+self.num_of_cols]
        self.video.append(backgournd_frame)
        self.save_video()
        self.play_video()
        return backgournd_frame
      
        
      
    def size(self):
        """Returns the video size in a list containing:
           [number_of_rows, number_of_columns]"""
        video_size = [self.num_of_rows, self.num_of_cols]
        print('number_of_rows = ', video_size[0], 'number_of_cols = ', video_size[1])
        
        return video_size

    
##########################################################33
    def new_frame(self, start_row, start_col):
        #"""Generates a single frame by extracting a portion of the background image."""
        # Same as init_video() but does not change the 
        # background image.
        
       #back_image = self.dic_of_back_imgs[back_name][start_row:start_row + self.num_of_rows, start_col:start_col+self.num_of_cols]
        back_image = self.video[0]
        self.video.append(back_image)
        
        return back_image
        
    def set_unfit_img(self, obj_name, start_row, start_col, start_row_new, end_row_new, start_col_new, end_col_new, clip_start_row, clip_end_row, clip_start_col, clip_end_col):
       # Set the new coordinates for unfit objects
                
       if start_row >= self.num_of_rows or start_col >= self.num_of_cols:
           print('Error: Object does not fit background.')
           
                
       if start_row < 0:
           clip_start_row = 0 - start_row
           start_row_new = 0
                
                      
       if start_row + self.dic_of_obj_imgs[obj_name][1].shape[0] >= self.num_of_rows and start_row < self.num_of_rows:
           clip_end_row = self.num_of_rows - start_row - 1 
           end_row_new = self.num_of_rows - 1
                
            
       if start_col < 0:
           clip_start_col = 0 - start_col
           start_col_new = 0
                
                
       if start_col + self.dic_of_obj_imgs[obj_name][1].shape[1] >= self.num_of_cols and start_col < self.num_of_cols:
           clip_end_col = self.num_of_cols - start_col - 1 
           end_col_new = self.num_of_cols - 1 
        
       return (start_row_new, end_row_new, start_col_new, end_col_new, 
                    clip_start_row, clip_end_row, clip_start_col, clip_end_col)
        
        
        
        
    def place_obj(self, obj_name, start_row, start_col):
        #print('place_obj:::obj_name = ', obj_name)
        self.obj_locs[obj_name]= [start_row, start_col]
        """Place image in the current video frame."""
        roi_obj_mask = self.dic_of_obj_imgs[obj_name][1]
        obj_img = self.dic_of_obj_imgs[obj_name][0]
        
        
        roi_obj = np.ones((roi_obj_mask.shape[0], roi_obj_mask.shape[1],3))
        roi_obj[:,:,0] = np.copy(roi_obj_mask * obj_img[:,:,0])
        roi_obj[:,:,1] = np.copy(roi_obj_mask * obj_img[:,:,1])
        roi_obj[:,:,2] = np.copy(roi_obj_mask * obj_img[:,:,2])
        
        mask_1 = np.ones((roi_obj_mask.shape[0], roi_obj_mask.shape[1]))
        roi_obj_mask_rev = mask_1 - roi_obj_mask
        
        
        
        
        #roi_obj = roi_obj_mask *
        if len(self.video) == 0:
            print('Warning: No frame.')
            
        else:
            back_obj_image = np.copy(self.video[-1])
        
        
            start_row_new = start_row
            end_row_new = start_row + roi_obj_mask.shape[0]
            start_col_new = start_col
            end_col_new = start_col + roi_obj_mask.shape[1]
       
        
            clip_start_row = 0
            clip_end_row = roi_obj_mask.shape[0]
            clip_start_col = 0
            clip_end_col = roi_obj_mask.shape[1]
        
            if start_row < 0 or start_col < 0 or (start_row + roi_obj_mask.shape[0]) >= self.num_of_rows or (start_col + roi_obj_mask.shape[1]) >= self.num_of_cols:
                #print('Warning: Object does not fit background.')
                (start_row_new, end_row_new, start_col_new, end_col_new,
             clip_start_row, clip_end_row, clip_start_col, clip_end_col)=self.set_unfit_img(obj_name, start_row, start_col, start_row_new, end_row_new, start_col_new, end_col_new, clip_start_row, clip_end_row, clip_start_col, clip_end_col)
                
            
            
                clip_obj_mask_rev = np.copy(roi_obj_mask_rev[clip_start_row:clip_end_row, clip_start_col: clip_end_col])
                clip_obj_mask = np.copy(roi_obj_mask[clip_start_row:clip_end_row, clip_start_col: clip_end_col])
                
                roi_back_clip = np.copy(np.ones((clip_obj_mask_rev.shape[0], clip_obj_mask_rev.shape[1],3)))
                roi_obj_clip = np.copy(obj_img[clip_start_row:clip_end_row, clip_start_col: clip_end_col])
                
                roi_back_clip [:,:,0] =np.copy( back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new][:,:,0] * clip_obj_mask_rev)
                roi_back_clip [:,:,1] =np.copy( back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new][:,:,1] * clip_obj_mask_rev)
                roi_back_clip [:,:,2] =np.copy( back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new][:,:,2] * clip_obj_mask_rev)
                
                roi_obj_clip [:,:,0] = np.copy(roi_obj_clip[:,:,0] * clip_obj_mask)
                roi_obj_clip [:,:,1] = np.copy(roi_obj_clip[:,:,1] * clip_obj_mask)
                roi_obj_clip [:,:,2] = np.copy(roi_obj_clip[:,:,2] * clip_obj_mask)
               
                back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new] = np.copy(roi_back_clip  + roi_obj_clip)
                #self.obj_locs[obj_name]= [start_row_new, start_col_new]
              
                
            else:  ##MASK *, FRAME +
                
                roi_back = np.ones((roi_obj.shape[0], roi_obj.shape[1],3))
                
                roi_back[:,:,0] = np.copy(back_obj_image[start_row:(start_row + roi_obj_mask.shape[0]), start_col:(start_col +roi_obj_mask.shape[1])][:,:,0] * roi_obj_mask_rev)
                roi_back[:,:,1] = np.copy(back_obj_image[start_row:(start_row + roi_obj_mask.shape[0]), start_col:(start_col +roi_obj_mask.shape[1])][:,:,1] * roi_obj_mask_rev)
                roi_back[:,:,2] = np.copy(back_obj_image[start_row:(start_row + roi_obj_mask.shape[0]), start_col:(start_col +roi_obj_mask.shape[1])][:,:,2] * roi_obj_mask_rev)
                back_obj_image[start_row:(start_row + roi_obj_mask.shape[0]), start_col:(start_col +roi_obj_mask.shape[1])] = np.copy(roi_back + roi_obj)
               
        
        # 1. Retrieve the last video frame else print a warning.
        # 2. Write the code to multiply the roi by the object img and place it 
        #    against the bakcground. You need to check if the image fits
        #    inside the background. If it does not fit, you must clip the object
        #    and save the part of the image that fits.
        # 3. Save the object location into a dictionary for the current video.
                #self.obj_locs[obj_name]= [start_row, start_col]
        # 4. Update the last video frame. 
            self.video[-1] = back_obj_image
            self.save_video()
            self.play_video()
    
            return back_obj_image
       
         
    def move_up(self, obj_name, row_motion):
        
        (start_row, start_col) = self.obj_locs[obj_name]                 
        return (start_row - row_motion, start_col)
    
    def move_down(self, obj_name, row_motion):
        
        (start_row, start_col) = self.obj_locs[obj_name]                 
        return (start_row + row_motion, start_col)
    
    def move_left(self, obj_name, col_motion):
        
        (start_row, start_col) = self.obj_locs[obj_name]                 
        return (start_row, start_col - col_motion)
    
    def move_right(self, obj_name, col_motion):
        
        (start_row, start_col) = self.obj_locs[obj_name]                 
        return (start_row, start_col + col_motion)
        
    
    def fw(self, time):
        obj_name =list(self.dic_of_obj_imgs.items())[-1][0]
        #print('obj_name = ', obj_name)
        for i in range (1, int(time * self.frame_ps)):
            
            self.new_frame(0, 0) 
            (start_row, start_col) = self.obj_locs[obj_name]
        
        
        
            direction = self.dic_of_obj_rot[obj_name][0]
            angle = self.dic_of_obj_rot[obj_name][1]
        
            if direction == 'r':
                if (abs(angle)//180) % 2 == 0:
                    new_direction = 'r'
                    new_angle = abs(angle) % 180
                else:
                    new_direction = 'l'
                    new_angle = 180 - abs(angle) % 180
                
            if direction == 'l':
                if (abs(angle)//180) % 2 == 0:
                    new_direction = 'l'
                    new_angle = abs(angle) % 180
                else:
                    new_direction = 'r'
                    new_angle = 180 - abs(angle) % 180
       
            #print ('new_direction= ',new_direction)
            #print ('new_ang = ', new_angle)
        
            if new_direction == 'r':
                if new_angle >= 0 and new_angle <= 90:
                    up_dis = int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    right_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
               
                    (start_row_1, start_col_1) =  self.move_up(obj_name, up_dis)
                    (start_row_2, start_col_2) = self.move_right(obj_name, right_dis)
               
                else:
                    down_dis = -1 * int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    right_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_down(obj_name, down_dis)
                    (start_row_2, start_col_2) =self.move_right(obj_name, right_dis)
                
            if new_direction == 'l':
                if new_angle >= 0 and new_angle <= 90:
                    up_dis = int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    left_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_up(obj_name, up_dis)
                    (start_row_2, start_col_2) =self.move_left(obj_name, left_dis)
                else:
                    down_dis = -1 * int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    left_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_down(obj_name, down_dis)
                    (start_row_2, start_col_2) =self.move_left(obj_name, left_dis)
                
            new_row = int(start_row_1)
            new_col = int(start_col_2)
        
            self.place_obj(obj_name, new_row, new_col)
            #self.save_video()
            #self.play_video()  
            #cv2.imshow('second frame', frame)
            #cv2.waitKey(10)               
        #return (frame, new_row, new_col)
    
        
        
        
        return (new_row, new_col)
    
    def bw(self, time):
        obj_name =list(self.dic_of_obj_imgs.items())[-1][0]
        #print('obj_name = ', obj_name)
        for i in range (1, int(time * self.frame_ps)):
            
            self.new_frame(0, 0) 
            (start_row, start_col) = self.obj_locs[obj_name]
        
        
        
            direction = self.dic_of_obj_rot[obj_name][0]
            angle = self.dic_of_obj_rot[obj_name][1]
        
            if direction == 'r':
                if (abs(angle)//180) % 2 == 0:
                    new_direction = 'r'
                    new_angle = abs(angle) % 180
                else:
                    new_direction = 'l'
                    new_angle = 180 - abs(angle) % 180
                
            if direction == 'l':
                if (abs(angle)//180) % 2 == 0:
                    new_direction = 'l'
                    new_angle = abs(angle) % 180
                else:
                    new_direction = 'r'
                    new_angle = 180 - abs(angle) % 180
       
            #print ('new_direction= ',new_direction)
            #print ('new_ang = ', new_angle)
        
            if new_direction == 'r':
                if new_angle >= 0 and new_angle <= 90:
                    down_dis = int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    left_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
               
                    (start_row_1, start_col_1) =  self.move_down(obj_name, down_dis)
                    (start_row_2, start_col_2) = self.move_left(obj_name, left_dis)
               
                else:
                    up_dis = -1 * int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    left_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_up(obj_name, up_dis)
                    (start_row_2, start_col_2) =self.move_left(obj_name, left_dis)
                
            if new_direction == 'l':
                if new_angle >= 0 and new_angle <= 90:
                    down_dis = int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    right_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_down(obj_name, down_dis)
                    (start_row_2, start_col_2) =self.move_right(obj_name, right_dis)
                else:
                    up_dis = -1 * int(self.px_frame) * math.cos(new_angle/180*math.pi)
                    right_dis = int(self.px_frame) * math.sin(new_angle/180*math.pi)
                    (start_row_1, start_col_1) = self.move_up(obj_name, up_dis)
                    (start_row_2, start_col_2) =self.move_right(obj_name, right_dis)
                
            new_row = int(start_row_1)
            new_col = int(start_col_2)
        
            self.place_obj(obj_name, new_row, new_col)
            #self.save_video()
            #self.play_video()  
            #cv2.imshow('second frame', frame)
            #cv2.waitKey(10)               
        return (new_row, new_col)
                
                
                
        
    
    
    
    
   
    def rt(self, rot_angle):
        obj_name =list(self.dic_of_obj_imgs.items())[-1][0]
        
        (start_row, start_col) = self.obj_locs[obj_name]
       
        if self.dic_of_obj_rot[obj_name][0] == 'r':
            new_angle = abs(self.dic_of_obj_rot[obj_name][1]) + rot_angle
        if self.dic_of_obj_rot[obj_name][0] == 'l':
            new_angle = -abs(self.dic_of_obj_rot[obj_name][1]) + rot_angle
       
        
        if new_angle >= 0:
            direction = 'r'
            
        if new_angle < 0:
            direction = 'l'
        
            
        rot_obj_name = obj_name + '_' + direction + '_' + str(abs(new_angle))
      
        self.add_obj_rot(rot_obj_name, direction, abs(new_angle))
        
        
        roi_obj_mask = np.copy(self.dic_of_obj_imgs[obj_name][1])
        obj_img = np.copy(self.dic_of_obj_imgs[obj_name][0])
        
       
        M = cv2.getRotationMatrix2D(((obj_img.shape[1])/2,(obj_img.shape[0])/2), -1*rot_angle,1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
       
        nW = int((roi_obj_mask.shape[0] * sin) + (roi_obj_mask.shape[1] * cos))
        nH = int((roi_obj_mask.shape[0] * cos) + (roi_obj_mask.shape[1] * sin))
        
        M[0, 2] += (nW / 2) - (obj_img.shape[1])/2
        M[1, 2] += (nH / 2) - (obj_img.shape[0])/2
        obj_rot = np.copy(cv2.warpAffine(obj_img, M,(nW, nH)))
        
        obj_rot_mask = np.copy(cv2.warpAffine(roi_obj_mask.astype(float), M,(nW, nH)))
        
        self.add_obj(rot_obj_name, obj_rot, obj_rot_mask)
        self.obj_locs[rot_obj_name]= [start_row, start_col]
        self.place_obj(rot_obj_name, start_row, start_col)
        
    
       # print('dir = ', direction)
        #print('new_angle = ', rot_angle)
        return rot_obj_name
    
    
    
    def lt(self, rot_angle):
        
        obj_name =list(self.dic_of_obj_imgs.items())[-1][0]
        (start_row, start_col) = self.obj_locs[obj_name]
        
        if self.dic_of_obj_rot[obj_name][0] == 'r':
            new_angle = abs(self.dic_of_obj_rot[obj_name][1]) - rot_angle
        if self.dic_of_obj_rot[obj_name][0] == 'l':
            new_angle = -abs(self.dic_of_obj_rot[obj_name][1]) - rot_angle
       
       
        if new_angle >= 0:
            direction = 'r'
            
        if new_angle < 0:
            direction = 'l'
        
            
        rot_obj_name = obj_name + '_' + direction + '_' + str(abs(new_angle))
        
        #print('dir = ', direction)
        #print('new_angle = ', abs(new_angle))
        
        self.add_obj_rot(rot_obj_name, direction, abs(new_angle))
        
        
        
        roi_obj_mask = np.copy(self.dic_of_obj_imgs[obj_name][1])
        obj_img = np.copy(self.dic_of_obj_imgs[obj_name][0])
        
       
        M = cv2.getRotationMatrix2D(((obj_img.shape[1])/2,(obj_img.shape[0])/2), rot_angle,1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        #print ('cos = ', cos)
        #print ('sin = ', sin)
       
        nW = int((roi_obj_mask.shape[0] * sin) + (roi_obj_mask.shape[1] * cos))
        nH = int((roi_obj_mask.shape[0] * cos) + (roi_obj_mask.shape[1] * sin))
        
        #print('nW = ', nH, 'nH = ',nH)
        M[0, 2] += (nW / 2) - (obj_img.shape[1])/2
        M[1, 2] += (nH / 2) - (obj_img.shape[0])/2
        
        #print('M = ', M)
        obj_rot = np.copy(cv2.warpAffine(obj_img, M,(nW, nH)))
        
        obj_rot_mask = np.copy(cv2.warpAffine(roi_obj_mask.astype(float), M,(nW, nH)))
        
        self.add_obj(rot_obj_name, obj_rot, obj_rot_mask)
        self.obj_locs[rot_obj_name]= [start_row, start_col]
        self.place_obj(rot_obj_name, start_row, start_col)
        #print('lt::obj_name = ', obj_name)
        return rot_obj_name
        
        
        
      
    def scale_obj(self, obj_name, row_center, col_center, scale_factor):
        """Place image at (row_center, col_center) enlarged by scale_factor."""
        
        roi_obj_mask = np.copy(self.dic_of_obj_imgs[obj_name][1])
       
        obj_img = np.copy(self.dic_of_obj_imgs[obj_name][0])
        
       
        res = cv2.resize(obj_img, (int(scale_factor* obj_img.shape[1]), int(scale_factor* obj_img.shape[0])))
        res_roi_mask = cv2.resize(roi_obj_mask.astype(float), (int(scale_factor* roi_obj_mask.shape[1]), int(scale_factor* roi_obj_mask.shape[0])))
       # (start_row, start_col) = self.obj_locs[obj_name]
        
    
        return (res, res_roi_mask)

    def play_video(self):
        """Plays the video on the screen."""
        
        cv2.imshow(self.video_name, self.video[-1])
        cv2.waitKey(1)
        
    def save_video(self):
        """Saves the video to a file."""
        self.out.write(self.video[-1])
        
    def beep(self, fq, t):
        winsound.Beep(fq, t)
        time.sleep(0.5)
    

        
    def beep1(self):
        Freq = 2500 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        self.beep(Freq,Dur)
        
    def beep2(self):
        Freq_1 = 500 # Set Frequency To 2500 Hertz
        Freq_2 = 1000 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        self.beep(Freq_1,Dur)
        self.beep(Freq_2,Dur)
        
    def beep3(self):
        Freq_1 = 500 # Set Frequency To 2500 Hertz
        Freq_2 = 1200 # Set Frequency To 2500 Hertz
        Freq_3 = 1500 # Set Frequency To 2500 Hertz
        Freq_4 = 1700 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        self.beep(Freq_1,Dur)
        self.beep(Freq_2,Dur)
        self.beep(Freq_3,Dur)
        self.beep(Freq_4,Dur)
        
    def beepLong(self):
        soo = 247
        do = 330 
        re = 370 
        mi = 415
        fa = 440 
        so = 494 
        la = 554
        ti = 622
        
        Dur = 250
        
        self.beep(mi,Dur)
        self.beep(mi,Dur)
        self.beep(fa,Dur)
        self.beep(so,Dur)
        self.beep(so,Dur)
        self.beep(fa,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(re,Dur)
        
        self.beep(mi,Dur)
        self.beep(mi,Dur)
        self.beep(fa,Dur)
        self.beep(so,Dur)
        self.beep(so,Dur)
        self.beep(fa,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(do,Dur)
        
        self.beep(re,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(fa,Dur)
        self.beep(mi,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(fa,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(soo,Dur)
        self.beep(soo,Dur)
        
        self.beep(mi,Dur)
        self.beep(mi,Dur)
        self.beep(fa,Dur)
        self.beep(so,Dur)
        self.beep(so,Dur)
        self.beep(fa,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(do,Dur)
        self.beep(re,Dur)
        self.beep(mi,Dur)
        self.beep(re,Dur)
        self.beep(do,Dur)
        self.beep(do,Dur)
        
        
        
        
    