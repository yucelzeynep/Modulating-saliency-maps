#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:52:06 2020

@author: zeynep
"""



import numpy as np

import pickle

from person import Person

import matplotlib.pyplot as plt
from importlib import reload

import preferences
reload(preferences)

import constants
reload(constants)

import time

def init():
    """
    This function initializes the 2D r2center array.
    1st dimension is participant and 2nd dimension is time.
    
    Not all participants have the same number of samples. I allow the number of 
    samples to be between 90 and 100. See below for reasons of this choice. 
    """
    r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype = \
    {}, {}, {}
    
    for age_range in constants.AGE_RANGES:
        r2center_2d_wrt_age[age_range] = []
            
    for m in constants.MOTIVATIONS:
          r2center_2d_wrt_motiv[m] = []
        
    for objtype in preferences.OBJECT_TYPES_INTEREST:
        r2center_2d_wrt_objtype[objtype] = []
        

    
    return r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype



def pad_fixation_array(r2center):
    """
    This function pads the fixation array to NMAX_FIXATIONS. 
    
    Since the fixation target is shown for 1080 msec (see constants),
    we expect to have roughly 97 samples over each fixation target.
    
    Nevertheless, the sensor does not always sample 90 samples in a sec. If the 
    participant blinks, looks away etc, we collect less than 90 samples per sec.

    If the number of samples between 90 and 100, we align the arrays by simply 
    padding with 0s. 
    
    There is actually some discrepancy here. Because the missing samples can 
    be missed at any point in time. But we assume that they are all missed at the 
    end. 
    
    It  does not matter so much because I take array with length 90 shortest. So
    even though things shift a little in time, they definitely do not shift more 
    than 0.07 sec. Actually, in most cases it is not more than 0.02-0.03 msec 
    """
    for i in range(preferences.NMAX_FIXATIONS - len(r2center)):
        r2center.append(0)
        
    return r2center

def get_r2center_medians_wrt_age(r2center_2d):
    """
    This function gets the median distance to center at target image overall, i.e.
    for all participants grouped by age, all motivations, all objtype
    """
    r2center_medians_wrt_age = {}
    
    for age_range in constants.AGE_RANGES:
        r2center_medians_wrt_age[age_range] = []
    
        """
        Number of time samples can be at most 100
        Because the target is shown for 1 min
        """
        
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """
            temp = [r2center[n] for r2center in r2center_2d[age_range]]
            r2center_medians_wrt_age[age_range].append(np.median(temp))
            

            
    return r2center_medians_wrt_age

def get_r2center_medians_wrt_motiv(r2center_2d_wrt_motiv):
    """
    This function gets the median distance to center at target image 
    for each motivation separately
    """
    
    r2center_medians_wrt_motiv = {}

    for m in constants.MOTIVATIONS:
        
        r2center_medians_wrt_motiv[m] = []
 
        r2center_2d = r2center_2d_wrt_motiv[m]
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """
            temp = [r2center[n] for r2center in r2center_2d]
            r2center_medians_wrt_motiv[m].append(np.median(temp))

        
    return r2center_medians_wrt_motiv

def get_r2center_medians_wrt_objtype(r2center_2d_wrt_objtype):
    """
    This function gets the median distance to center at target image 
    for each objtype separately
    """
    r2center_medians_wrt_objtype = {}
    
    for o in preferences.OBJECT_TYPES_INTEREST:
        r2center_medians_wrt_objtype[o] = []
        r2center_2d = r2center_2d_wrt_objtype[o]
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """            
            temp = [r2center[n] for r2center in r2center_2d]
            r2center_medians_wrt_objtype[o].append(np.median(temp))
     
    
    return r2center_medians_wrt_objtype




def plot_r2center_wrt_age(r2center_medians_wrt_age):
    
    fr, axr = plt.subplots()
    axr.set_xlim([0,100])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for age_range in constants.AGE_RANGES:
            
        plt.plot(r2center_medians_wrt_age[age_range],  label=age_range+'_medians')
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('N pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
        
        data = np.array([range(len(r2center_medians_wrt_age[age_range])), r2center_medians_wrt_age[age_range]])
        data = data.T
        #here you transpose your data, so to have it in two columns        
        datafile_path = 'r2center_medians_wrt_age_' + age_range + '.txt'
#        with open(datafile_path, 'w+') as datafile_id:        
#            np.savetxt(datafile_id, data, fmt=['%d','%d'])
    
    plt.legend()
    plt.show()
    
    
def plot_r2center_wrt_motiv(r2center_medians_wrt_motiv):
    
    fr, axr = plt.subplots()
    axr.set_xlim([0,100])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    
    for m in constants.MOTIVATIONS:
        
        plt.plot(r2center_medians_wrt_motiv[m], label=m)
        

        
        #######################################################################
        # save wrt motiv
        #
        data = np.array([range(len(r2center_medians_wrt_motiv[m])),\
                         r2center_medians_wrt_motiv[m]\
                         ])
        data = data.T
        #here you transpose your data, so to have it in two columns        
        datafile_path = 'r2center_medians_wrt_motiv_'  + m + '.txt'
#        with open(datafile_path, 'w+') as datafile_id:        
#            np.savetxt(datafile_id, data, fmt=['%d','%d'])
        
     
    plt.xlabel('time (0-1 sec)')
    plt.ylabel('N pixels to target center')
    plt.grid(linestyle='--', linewidth=1)
    
    plt.legend()
    plt.show()
        
def plot_r2center_wrt_objtype(r2center_medians_wrt_objtype):
    
    fr, axr = plt.subplots()
    axr.set_xlim([0,100])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for o in preferences.OBJECT_TYPES_INTEREST:
        
        
        plt.plot(r2center_medians_wrt_objtype[o], label=o.split('_')[-1][0:3])

    
        #######################################################################
        # save wrt objtype
        #
        data = np.array([range(len(r2center_medians_wrt_objtype[o])),\
                         r2center_medians_wrt_objtype[o]\
                         ])
        data = data.T
        #here you transpose your data, so to have it in two columns        
        datafile_path = 'r2center_medians_wrt_objtype_'  + o.split('_')[-1][0:3] + '.txt'
#        with open(datafile_path, 'w+') as datafile_id:        
#            np.savetxt(datafile_id, data, fmt=['%d','%d'])

    plt.xlabel('time (0-1 sec)')
    plt.ylabel('n pixels to target center')
    plt.grid(linestyle='--', linewidth=1)
        
    plt.legend()
    plt.show()
    

    
    
    
if __name__ == "__main__":
    
    start_time = time.time()
    
    #fig_tools.prep_figs()

    r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype,\
    = init()
    
    """
    This array collects all r2center for all participants (in some cases for 
    certain mitivations or objtype).
    
    The condition is that there has to be 90 to 100 samples
    Otherwise either the target is the initial target (displayed 2.5sec)
    or there are too many missing samples (user looks away, closes eyes etc)
    
    """


    for ppp, participant in enumerate( constants.PARTICIPANTS ):
        
        fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
        
            for objtype in preferences.OBJECT_TYPES_INTEREST:
                for kk, (image_fname, image_fixations) in \
                    enumerate(zip(person.image[objtype]['image_fnames'], \
                        person.image[objtype]['image_fixations'])):
                        
                        if image_fname == constants.TARGET_IMAGE_FNAME:
                            # this is a target image
                            
                            if (len(image_fixations) >= preferences.NMIN_FIXATIONS \
                                and \
                                len(image_fixations) <= preferences.NMAX_FIXATIONS):
                                # this target is displayed 1 sec and the person
                                # watched it long enough 
                                r2center = []
                                
                                for fixation in image_fixations:
                                    fixation_time = fixation[2] # in sec
                                    
                                    r =  np.sqrt(\
                                                (constants.CENTER_DOT_PX - fixation[0])**2 + \
                                                (constants.CENTER_DOT_PY - fixation[1])**2 \
                                                )
                                
                                    r2center.append( r )
                                    
                                """
                                Pad the array to lenth of 100 (with 0s)
                                append to corresponding arrays
                                """
                                r2center = pad_fixation_array(r2center)
                                
                                # overall array
                                r2center_2d_wrt_age[person.age_range].append(r2center)
                                
                                # by motiv
                                r2center_2d_wrt_motiv[person.motiv].append(r2center)
                                
                                # by objtype 
                                r2center_2d_wrt_objtype[objtype].append(r2center)
                                
        
                                
                                
                                    
    r2center_medians_wrt_age = get_r2center_medians_wrt_age(r2center_2d_wrt_age)
    r2center_medians_wrt_motiv = get_r2center_medians_wrt_motiv(r2center_2d_wrt_motiv)
    r2center_medians_wrt_objtype = get_r2center_medians_wrt_objtype(r2center_2d_wrt_objtype)
        
#    ###########################################################################
#    fname = 'r2center_medians_v2.pkl'
#    with open(str(fname), 'wb') as f:
#        pickle.dump([r2center_2d_wrt_age,\
#                     r2center_2d_wrt_motiv,\
#                     r2center_2d_wrt_objtype], f, pickle.HIGHEST_PROTOCOL)
    ###########################################################################
    fname = 'r2center_medians_v2.pkl'
    with open(fname,'rb') as f:
        [r2center_2d_wrt_age,\
                     r2center_2d_wrt_motiv,\
                     r2center_2d_wrt_objtype] = pickle.load(f)    
        
    plot_r2center_wrt_age(r2center_medians_wrt_age)
    plot_r2center_wrt_motiv(r2center_medians_wrt_motiv)
    plot_r2center_wrt_objtype(r2center_medians_wrt_objtype)
    
    
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)
