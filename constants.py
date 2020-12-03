#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:28:18 2020

@author: zeynep
"""

IMAGE_PATH = '../../data_to_release/images/'
RAW_GAZE_PATH = '../data_to_release/gaze/'

AGE_FILE_PATH = '../data/ages.txt'
FAMILIARITY_PATH = '../data/familiarity_ratings/'

ANNOTATION_POLYGON_FUNCTIONAL_DIR = '../data/polygons/functional_coder2/'
ANNOTATION_POLYGON_MANIPULATIVE_DIR = '../data/polygons/manipulative_coder2/'

OUTPUT_DIR = '../data/pickled/'
INPUT_DIR = '../data/pickled/'

AGE_RANGES = {
'young',\
'old'\
}


# f: freeview, u: use, p: push
MOTIVATIONS =\
{'f',\
 'u',\
 'p'\
 }


OBJECT_TYPES = [
'00_interaction_graspable_functional_manipulative_grip',\
'01_interaction_graspable_functional_manipulative_handle',\
'02_interaction_graspable_functional_only',\
'03_interaction_graspable_nontool',\
]


# functional (handle/grip), manipulative (end effector), neither
OBJECT_PARTS = [
'func',\
'manip',\
'neither'\
]

"""
The participants watched the images of object types in thesame order
as follows
"""
OBJECT_TYPE_VS_TIMELOG = {\
'0':'00_interaction_graspable_functional_manipulative_grip',\
'1':'01_interaction_graspable_functional_manipulative_handle',\
'2':'02_interaction_graspable_functional_only',\
'3':'03_interaction_graspable_nontool'
}


PARTICIPANTS = [\
'2020_01_06_09_f',\
'2020_01_06_10_f',\
'2020_01_06_13_f',\
'2020_01_06_15_f',\
'2020_01_07_10_f',\
'2020_01_07_14_f',\
'2020_01_07_15_u',\
'2020_01_08_10_u',\
'2020_01_08_13_u',\
'2020_01_08_14_u'\
'2020_01_08_15_u',\
'2020_01_09_10_u',\
'2020_01_09_13_p',\
'2020_01_09_14_p',\
'2020_01_10_09_p',\
'2020_01_14_10_p',\
'2020_01_14_13_p',\
'2020_01_15_14_p',\
'2020_01_16_14_f',\
'2020_01_16_14_f',\
'2020_01_16_15_f',\
'2020_01_20_12_u',\
'2020_01_20_12_u',\
'2020_01_20_12_u',\
'2020_01_20_13_p',\
'2020_01_20_13_p',\
'2020_01_20_13_p',\
'2020_01_20_14_f',\
'2020_01_20_14_p',\
'2020_01_20_14_u',\
'2020_01_20_15_u'\
]



# I display the cropped images in full screen mode on a screen with the 
# following resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

IMAGE_DUR = 2500 # display duration of each image (in msec)
TARGET_DUR = 1080 # display duration of target screen (in msec)
BLANK_DUR = 500 # display duration of blank screen (in msec)

TARGET_IMAGE_FNAME = 'target_v4.jpeg' # fixation target image file name
BLANK_IMAGE_FNAME = 'blank.jpeg'

CENTER_DOT_PX = 960
CENTER_DOT_PY = 540

