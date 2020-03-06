#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:54:20 2020

@author: zeynep
"""
import sys

sys.path.insert(0, '../') # for constants, and preferences

import numpy as np 
import file_tools as file_tools

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
                              file_tools.find_file_in_path('*.jpeg', \
                              constants.IMAGE_PATH + object_type))
        
        for image_path in image_paths:
            myobject = MyObject()
            myobject.builder(image_path)
            

    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

