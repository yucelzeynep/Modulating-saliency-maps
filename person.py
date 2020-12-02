#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:47:38 2020

@author: zeynep

This file contains the description of the class Person. This data structure 
contains information, which is **specific** to participants. Namely, their 
intrinsic properties (e.g. age, motivation) as well as experimatl data (e.g. gaze 
samples, fixation maps).
"""

import numpy as np
import pickle
import copy

import cv2

import tools_file as ftools
import saliency_tools as stools
import tools_fig as tools_fig

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

class Person():
    """
    Person is a collection of information on a *single* participant
    It involves all image names and gaze data as well as age, motivation etc.
    """
    def __init__(self):
        self.name = []
        self.age = []  
        self.age_range = []
        self.motiv = []        
        
        self.image = {}
        
        for object_type in constants.OBJECT_TYPES:
            """
            Here, I need to keep several structures as below, since some images 
            (fixation target and blank screen) are displayed multiple times, so I cannot use 
            simply their names as dictionary keys.
            
            For initializing fixation map, I use screen_height and screen_width,
            which are actually the same as image_height and image_width,
            since I display the images in full screen.
            But this is not necessarily true for any experiment (preliminaries were different).
            To highlight that, here I use screen params
            """
            self.image[object_type] = {'image_fnames':[],\
                                      'image_fixations': [],
                                      'fmaps': [],\
                                      'familiarities': []}
            
        
    def builder(self, participant):
        """
        Fill in the variables
        """
        self.get_name(participant)
        self.load_age(participant)
        self.get_motiv(participant)
        self.load_image_related(participant)
        self.build_fmap()
        self.load_familiarities(participant)
                
#        # save person object
#        fpath = constants.OUTPUT_DIR + 'person/' + participant + '.pkl'
#        with open(str(fpath), 'wb') as f:
#            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

       
        
    def get_name(self, participant):
        """
        This function picks up the last name of the participant
        
        This information is at 5th place in the parsed string
        """
        self.name =  participant.split("_")[5]
        
    def load_age(self, participant):
        """
        This function loads the age and sets the age_range.
        
        Anyone younger than 30 has an age range of young,
        and others have an age_range of elderly (old).
        
        The file with ages is not provided in the repository due to privacy 
        concerns.
        """
        age_list = np.genfromtxt(constants.AGE_FILE_PATH, delimiter='\t', dtype='U')
        for age_line in age_list:
            if age_line[0] == participant:
                self.age = int(age_line[1] )
                if self.age < 30:
                    self.age_range = 'young'
                else:
                    self.age_range = 'old'
        
    def get_motiv(self, participant):
        """
        This information is at the 0th letter of at 4th place in the parsed string
        
        Motivation possibilities together with their abbreviations are:
            f: freeview
            u: use
            p: push
            
        """
        m = participant.split("_")[4][0] # 4th string 0th letter 
        self.motiv = m
        
    def load_image_related(self,participant):
        """
        This field does not contain the image itself (i.e. jpeg file) but gaze 
        information of the participant over the images.
        
        The image variable has the following shape:
            image[object_type][image_fnames]
            image[object_type][image_fixations]
            image[object_type][fmaps]
            image[object_type][familiarities]
            
        I need to keep several structures as above, since some images 
        (fixation target and blank screen) are displayed multiple times, so I cannot use simply
        their names as dictionary keys.
            
        Into fixations, I copy the gaze as it is logged i.e.:
            px py t
        px and py are x- and y-coordinates, and t is timestamp in sec (with
        3 digits after decimal point, so millisec information is also there)
        """
    
        gaze_path = constants.RAW_GAZE_PATH + participant + '/' 
        gaze_fname = ftools.find_file_in_path('*_gaze.txt', gaze_path)[0]
        gaze_totdur = np.loadtxt(gaze_fname) # gaze over all viewing sessions
        
        """
        Timelog has the following structure:
            image_fname t0 tf
        
        Here, t0 is the instant, at which the image is displayed and tf is the 
        instant, at which it is removed. 
        """
        timelog_fnames = np.sort(ftools.find_file_in_path('*_timelog.txt', gaze_path))     
                
        for t, timelog_fname in enumerate(timelog_fnames):
            timelog = np.genfromtxt(timelog_fname,\
                             dtype=None, delimiter='\t',encoding='ASCII', \
                             names=('image_fname', 't0', 'tf'))

            object_type = constants.OBJECT_TYPE_VS_TIMELOG[str(t)]
            
            for timewindow in timelog:
                image_fname = timewindow[0]
                t0 = timewindow[1]
                tf = timewindow[2]
                
     
                self.image[object_type]['image_fnames'].append(image_fname)
                 
                temp_fixations = [] # gaze over a single image
                # append to array one by one
                for fixation in gaze_totdur:
                    fixation_time = fixation[2] # this is in sec (ie ss.mmm)
                    if fixation_time*1000 >t0 and fixation_time*1000 <= tf:
                        temp_fixations.append(fixation)
                        
                self.image[object_type]['image_fixations'].append(temp_fixations)
                    
              
    
    def build_fmap(self):
        """
        Build fixation maps (fmap). 
        """
          
        for object_type in preferences.OBJECT_TYPES_INTEREST:
            for image_fname, image_fixation in zip(\
                                                   self.image[object_type]['image_fnames'], \
                                                   self.image[object_type]['image_fixations'] ):
                                
                if image_fname != constants.TARGET_IMAGE_FNAME and\
                image_fname != constants.BLANK_IMAGE_FNAME:
                    
                    
                    # from discrete gaze samples to a distribution
                    fmap = stools.Fixpos2Densemap(image_fixation, \
                                                          constants.IMAGE_WIDTH, \
                                                          constants.IMAGE_HEIGHT)
                      
                    """
                    This part is for displaying image_orig, fixations and fmap 
                    Just for keeping track of what is happening
                    """
                    # load myobject and image_orig
                    myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                        
                    # image with fixations over image_orig
                    image_temp = cv2.merge([myobject.image_orig, myobject.image_orig, myobject.image_orig])
                    for g in image_fixation:
                        cv2.circle(image_temp, (int(g[0]), int(g[1])), 5, (10,255,25), -1)
                        
                    tools_fig.scale_and_display(myobject.image_orig, 'image_orig')
                    tools_fig.scale_and_display(image_temp, 'image_with_fixations')
                    tools_fig.scale_and_display(fmap, 'fmap')               
                            
                else:
                    # for target and blank image, I do not care about fmaps
                    fmap = []
                    
                self.image[object_type]['fmaps'].append(fmap)
               
                
    def load_familiarities(self, participant):
        """
        Load familiarities from the txt files.

	The participants rated their familiarity with the objects on a scale from 1 to 5.
	In  particular, we gave the following options:
        
        A familiarity of -1 indicates that no value is entered. In this case:
            1. The image is fixation target or blank screen.
            2. The participant missed to fill in that page of the questionnaire and did not write any scores 
            (Out of all 31 participants, 1 person failed to fill in one page of the questionnaire. 
		Her name is not released due to privacy concerns)

        Note that I did not collect familiarity score for the objects in the
        test run, because it is not relevant for the experiment. 
        Therefore, it catches the exception for these images.
        """
        fname = constants.FAMILIARITY_PATH + participant + '_familiarity.txt'
        scores = np.genfromtxt(fname, delimiter='\t', dtype='U')
        temp_names = [ l[0] for l in scores ]
        temp_scores = [ l[1] for l in scores ]
        
        
        for object_type in constants.OBJECT_TYPES:
            temp = []
            for image_fname in self.image[object_type]['image_fnames']:
                if image_fname != constants.BLANK_IMAGE_FNAME and\
                image_fname != constants.TARGET_IMAGE_FNAME:
                    
                    try:
                        ind = temp_names.index(image_fname)
                        temp.append(int (temp_scores[ind]) )
                    except ValueError:
                        print('Image fname is not found {}'.format((image_fname)))
                        temp.append( -1 )
                else:
                    temp.append( -1 )
                    
                    
            self.image[object_type]['familiarities'] = \
            copy.deepcopy( temp )
        
            #print('{} {}'.format(object_type, self.image[object_type]['familiarities']))
        
             
