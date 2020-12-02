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


# functional (handle/grip), manipulative (end effector), neither
OBJECT_PARTS = [
'func',\
'manip',\
'neither'\
]

OBJECT_TYPES = [
'00_interaction_graspable_functional_manipulative_grip',\
'01_interaction_graspable_functional_manipulative_handle',\
'02_interaction_graspable_functional_only',\
'03_interaction_graspable_nontool',\
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
'2020_01_06_09_fa_nakatsuka',\
'2020_01_06_10_fa_shimoda',\
'2020_01_06_13_fv_kuosaki',\
'2020_01_06_15_fv_koyoo',\
'2020_01_07_10_fav_haruna',\
'2020_01_07_14_fav_yamamoto',\
'2020_01_07_15_ua_shimoda',\
'2020_01_08_10_ua_sakai',\
'2020_01_08_13_uv_takeda',\
'2020_01_08_14_uv_yokoyama',\
'2020_01_08_15_uav_onoda',\
'2020_01_09_10_uav_hiraide',\
'2020_01_09_13_pa_otaka',\
'2020_01_09_14_pa_satoyama',\
'2020_01_10_09_pv_furukawa',\
'2020_01_14_10_pv_gotou',\
'2020_01_14_13_pav_oonishi',\
'2020_01_15_14_pav_katagi',\
'2020_01_16_14_f_nasu',\
'2020_01_16_14_f_shimada',\
'2020_01_16_15_f_parisa',\
'2020_01_20_12_u_ikemoto',\
'2020_01_20_12_u_inoshita',\
'2020_01_20_12_u_tanaka',\
'2020_01_20_13_p_kasagi',\
'2020_01_20_13_p_nakahara',\
'2020_01_20_13_p_nishikawa',\
'2020_01_20_14_f_saito',\
'2020_01_20_14_p_seto',\
'2020_01_20_14_u_kanehira',\
'2020_01_20_15_u_korenaga'\
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


# f: freeview, u: use, p: push
MOTIVATIONS =\
{'f',\
 'u',\
 'p'\
 }

T_PROBES = ['t0', 'tm', 'tf']

AGE_RANGES = {
'young',\
'old'\
}
