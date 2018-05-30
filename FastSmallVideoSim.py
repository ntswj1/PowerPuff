# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:51:20 2017

@author: Wenjing Shi
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

class FastSmallVideoSim:
    """Fast digital video simulator for small videos that fit in memory.
       Usage example: ? Provide usage example here.
    """ 
    # Initialize the dictionary of background and object images.
    dic_of_back_imgs = {}  
    dic_of_obj_imgs  = {}
    center = {}
    new_obj_name = 0
    #####################################
    def __init__(self, video_name, num_of_rows, num_of_cols):
        """Constructor that stores video name and video size."""
        self.video_name = video_name
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.video      = []
        self.obj_locs   = {}
        self.out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 35, (self.num_of_cols, self.num_of_rows))
        print('(self.num_of_cols, self.num_of_rows) = ', self.num_of_cols, self.num_of_rows)
        print("(Initializing {})".format(self.video_name))
        
    def add_back(self, back_name, back_img):
        """Adds a background image for sharing."""
        self.dic_of_back_imgs[back_name] = back_img
         
        
    def add_obj(self, obj_name, obj_img, obj_roi):
        """Adds an object image for sharing."""
        obj = (obj_img, obj_roi)
        self.dic_of_obj_imgs[obj_name] = obj 
        
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
        #self.save_video()
        #self.play_video()
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
        
        
        
        
        
    def place_mask_img(self, obj_name, mask_img, start_row, start_col):
        # Alomost same as place_obj(), but place mask at first, and then place objects.
        
        if len(self.video) == 0:
            print('Warning: No frame.')
            
        else:
            back_obj_image = np.copy(self.video[-1])
            start_row_new = start_row
            end_row_new = start_row + self.dic_of_obj_imgs[obj_name][1].shape[0]
            start_col_new = start_col
            end_col_new = start_col + self.dic_of_obj_imgs[obj_name][1].shape[1]
       
        
            clip_start_row = 0
            clip_end_row = self.dic_of_obj_imgs[obj_name][1].shape[0]
            clip_start_col = 0
            clip_end_col = self.dic_of_obj_imgs[obj_name][1].shape[1]
            
            
            
                
            if start_row < 0 or start_col < 0 or (start_row + self.dic_of_obj_imgs[obj_name][1].shape[0]) >= self.num_of_rows or (start_col + self.dic_of_obj_imgs[obj_name][1].shape[1]) >= self.num_of_cols:
                print('Warning: Object does not fit background.')
                (start_row_new, end_row_new, start_col_new, end_col_new,
             clip_start_row, clip_end_row, clip_start_col, clip_end_col)=self.set_unfit_img(obj_name, start_row, start_col, start_row_new, end_row_new, start_col_new, end_col_new, clip_start_row, clip_end_row, clip_start_col, clip_end_col)
            
                clip_obj = self.dic_of_obj_imgs[obj_name][1][clip_start_row:clip_end_row, clip_start_col: clip_end_col]
                clip_mask = mask_img[clip_start_row:clip_end_row, clip_start_col: clip_end_col]
            
                back_mask =  np.copy(back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new])
                
                back_mask = back_mask * clip_mask
                
                back_mask = back_mask + clip_obj
               
                
                #plt.imshow(back_mask)
                #plt.show()
                
                back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new]=back_mask
            
            else:  ##MASK *, FRAME +
                back_mask = np.copy(back_obj_image[start_row_new:end_row_new, start_col_new:end_col_new])
                back_mask = back_mask * mask_img
                
                back_mask = back_mask + self.dic_of_obj_imgs[obj_name][1]
                
                #plt.imshow(back_mask)
                #plt.show()
                
                back_obj_image[start_row:(start_row + self.dic_of_obj_imgs[obj_name][1].shape[0]), start_col:(start_col +self.dic_of_obj_imgs[obj_name][1].shape[1])]=back_mask
        
        # 1. Retrieve the last video frame else print a warning.
        # 2. Write the code to multiply the roi by the object img and place it 
        #    against the bakcground. You need to check if the image fits
        #    inside the background. If it does not fit, you must clip the object
        #    and save the part of the image that fits.
        # 3. Save the object location into a dictionary for the current video.
            self.obj_locs[obj_name]= [start_row, start_col]
        # 4. Update the last video frame. 
            self.video[-1] = back_obj_image
        
            return back_obj_image
        
        
        
    def place_obj(self, obj_name, start_row, start_col):
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
            
        
            return back_obj_image
       
         
                       
        
    def move_obj(self, obj_name, row_motion, col_motion):
        """Place image in the current video frame by moving it using
           row_motion and col_motion pixels"""
       # 1. Retrieve the last saved location of the object from previous frame.
        # 2. Call place_obj() with new coordinates.
           
       
        (start_row, start_col) = self.obj_locs[obj_name]
        
       
        
        #new_place = self.place_obj(obj_name, start_row + row_motion, start_row + row_motion)
        #return new_place
        return (start_row + row_motion, start_col + col_motion)
   
            
        
    def rot_obj(self, obj_name, row_center, col_center, rot_angle):
        """Place image at (row_center, col_center) rotated by rot_angle"""
        
        roi_obj_mask = np.copy(self.dic_of_obj_imgs[obj_name][1])
        obj_img = np.copy(self.dic_of_obj_imgs[obj_name][0])
        #print(roi_obj_mask.shape[:2])
        new_obj_name = obj_name + '_rot_' + str(rot_angle)
        
        #self.new_obj_name = self.new_obj_name + 1
        
        y = roi_obj_mask.shape[0]/2
        x = roi_obj_mask.shape[1]/2
        
       
       
        ############################################################################
        M = cv2.getRotationMatrix2D(( x, y), rot_angle,1)
        #print(M)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        
       
        nW = int((2*y * sin) + (2*x * cos))  # new col
        nH = int((2*y * cos) + (2*x * sin))  # new row
        
        M[0, 2] += (nW / 2) -  x
        M[1, 2] += (nH / 2) -  y
        
        obj_rot = np.copy(cv2.warpAffine(obj_img, M,(nW, nH)))
        obj_rot_mask = np.copy(cv2.warpAffine(roi_obj_mask.astype(float), M,(nW, nH)))
       
        new_start_row_delta = (nH - 2*y)/2
        new_start_col_delta = (nW - 2*x)/2
        
        
        
       # plt.imshow(obj_rot)
       # plt.show()
       # print(obj_rot.shape[:2])
        ###############################################################################
        
        #obj_rot = np.copy(cv2.warpAffine(obj_img, M,(roi_obj_mask.shape[1], roi_obj_mask.shape[0])))
        #obj_rot_mask = np.copy(cv2.warpAffine(roi_obj_mask.astype(float), M,(roi_obj_mask.shape[1], roi_obj_mask.shape[0])))
       
        self.add_obj(new_obj_name, obj_rot, obj_rot_mask) 
        return (new_obj_name, new_start_row_delta, new_start_col_delta)
    
      
    def scale_obj(self, obj_name, row_center, col_center, scale_factor):
        """Place image at (row_center, col_center) enlarged by scale_factor."""
        
        roi_obj_mask = np.copy(self.dic_of_obj_imgs[obj_name][1])
       
        obj_img = np.copy(self.dic_of_obj_imgs[obj_name][0])
        
        #print(roi_obj_mask.shape, obj_img.shape)
        
        #self.new_obj_name = self.new_obj_name + 1
        new_obj_name = obj_name + '_scl_' + str(scale_factor)
        res = cv2.resize(obj_img, (int(scale_factor* obj_img.shape[1]), int(scale_factor* obj_img.shape[0])))
        res_roi_mask = cv2.resize(roi_obj_mask.astype(float), (int(scale_factor* roi_obj_mask.shape[1]), int(scale_factor* roi_obj_mask.shape[0])))
       # (start_row, start_col) = self.obj_locs[obj_name]
       
        if scale_factor < 0:
            new_start_row_delta = -1*((scale_factor-1)* obj_img.shape[0]/2)
            new_start_col_delta = -1*((scale_factor-1)* obj_img.shape[1]/2)
        else:
            new_start_row_delta = ((scale_factor-1)* obj_img.shape[0]/2)
            new_start_col_delta = ((scale_factor-1)* obj_img.shape[1]/2)
        
        self.add_obj(new_obj_name, res, res_roi_mask) 
       # plt.imshow(res)
        #plt.show()
        #print(res.shape[:2])
        #self.center[self.new_obj_name] = res.shape[:2]
        return (new_obj_name, new_start_row_delta, new_start_col_delta)

    def play_video(self):
        """Plays the video on the screen."""
        for i in range (0, len(self.video)):
            cv2.imshow(self.video_name, self.video[i])
            cv2.waitKey(10)
        
    def save_video(self):
        """Saves the video to a file."""
        for i in range (0, len(self.video)):
            self.out.write(self.video[i])
        
        
        
    