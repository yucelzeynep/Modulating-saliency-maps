#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:54:20 2020

@author: zeynep

This function loads the raw data concerning gaze samples and object images and 
arranges them as dictionary variables and prepares them for later processing. 
"""
import numpy as np 
import tools_file as tools_file

from person import Person
from myobject import MyObject

from importlib import reload
import constants
reload(constants)

import preferences
reload(preferences)

import time

if __name__ == "__main__":

    start_time = time.time()
    
    for participant in constants.PARTICIPANTS:
        person = Person()
        person.builder(participant)
   
    for object_type in preferences.OBJECT_TYPES_INTEREST:
        image_paths = np.sort(\
                              tools_file.find_file_in_path('*.jpeg', \
                              constants.IMAGE_PATH + object_type))
        
        for image_path in image_paths:
            myobject = MyObject()
            myobject.builder(image_path)
            

    """
    Time elapsed  10099.60 sec
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

