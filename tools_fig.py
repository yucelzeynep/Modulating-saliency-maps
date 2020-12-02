#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:20:19 2020

@author: zeynep
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)


def plot_r2center_overall(r2center_stats_overall):
    
    fr, axr = plt.subplots()
    axt = axr.twinx()  # right side is theta axis
    axr.set_xlim([0,100])
    axr.set_ylim([0,400])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for age_range in constants.AGE_RANGES:
            
        p = plt.plot(r2center_stats_overall[age_range]['means'], label=age_range+'_means')
        plt.plot(r2center_stats_overall[age_range]['medians'],  '--', color=p[0].get_color(), label=age_range+'_medians')
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('N pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
    
    plt.legend()
    plt.show()
    
    
def plot_r2center_wrt_motiv(r2center_medians_wrt_motiv):
    
    fr, axr = plt.subplots()
    axt = axr.twinx()  # right side is theta axis
    axr.set_xlim([0,100])
    axr.set_ylim([0,400])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    
    for m in constants.MOTIVATIONS:
        
        p = plt.plot(r2center_medians_wrt_motiv['old'][m]['means'], label='old'+'_'+m+'_means')
        plt.plot(r2center_medians_wrt_motiv['young'][m]['means'], '--', color=p[0].get_color(),)
        
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('N pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
    
    plt.legend()
    plt.show()
        
def plot_r2center_wrt_object_types(r2center_medians_wrt_object_types):
    
    fr, axr = plt.subplots()
    axt = axr.twinx()  # right side is theta axis
    axr.set_xlim([0,100])
    axr.set_ylim([0,400])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for o in constants.OBJECT_TYPES:
        p = plt.plot(r2center_medians_wrt_object_types['means'][o], label=o)
        plt.plot(r2center_medians_wrt_object_types['medians'][o], '--', color=p[0].get_color())
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('n pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
        
    plt.legend()
    plt.show()
    

        
def scale_and_display(image, window_title):
    """
    Scale a (color or gray level) image and display it
    See preferences for SCALE_PERCENT
    """
    # for scaling
    if len(image.shape) == 2:
        # gray
        H, W = image.shape
    else:
        # color
        H, W, _ = image.shape
        
    scaled_height = int( H * preferences.SCALE_PERCENT ) 
    scaled_width = int( W  * preferences.SCALE_PERCENT )
    DIM = (scaled_width, scaled_height)
    
    image_temp = cv2.resize(image, DIM, interpolation = cv2.INTER_AREA) 
    cv2.imshow(window_title, image_temp)
    
    cv2.waitKey(2)

    