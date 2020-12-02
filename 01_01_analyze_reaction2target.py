#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:52:06 2020

@author: zeynep

This function carries out the analysis necessary to derive the results reported 
in the section "Reaction to fixation target" in the manuscript, and specifically, 
the Fig 2 of the manuscript. 
"""

import numpy as np
import pickle
import tools_reaction as rtools
import tools_fig as ftools

from importlib import reload

import preferences
reload(preferences)

import constants
reload(constants)

import time
    
    
if __name__ == "__main__":
    
    start_time = time.time()
    
    r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype,\
    = rtools.init_r2center_2D()
    
    """
    This array collects all r2center in arrays arranged wrt each single intrinsic 
    or extrinsic feature, or for each combination of those.
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
                            # this is a fixation target image
                            
                            if (len(image_fixations) >= preferences.NMIN_FIXATIONS \
                                and \
                                len(image_fixations) <= preferences.NMAX_FIXATIONS):
                                # This image is displayed about 1 sec and the person
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
                                Pad the array to length of 100 (with 0s)
                                append to corresponding arrays
                                """
                                r2center = rtools.pad_fixation_array(r2center)
                                
                                # by age
                                r2center_2d_wrt_age[person.age_range].append(r2center)
                                
                                # by motiv
                                r2center_2d_wrt_motiv[person.motiv].append(r2center)
                                
                                # by objtype 
                                r2center_2d_wrt_objtype[objtype].append(r2center)
                                
        
                                    
    r2center_medians_wrt_age = rtools.get_r2center_medians_wrt_age(r2center_2d_wrt_age)
    r2center_medians_wrt_motiv = rtools.get_r2center_medians_wrt_motiv(r2center_2d_wrt_motiv)
    r2center_medians_wrt_objtype = rtools.get_r2center_medians_wrt_objtype(r2center_2d_wrt_objtype)
        
    ###########################################################################
    fname = 'pkl_files/r2center_medians_v2.pkl'
    with open(str(fname), 'wb') as f:
        pickle.dump([r2center_2d_wrt_age,\
                     r2center_2d_wrt_motiv,\
                     r2center_2d_wrt_objtype], f, pickle.HIGHEST_PROTOCOL)
    ###########################################################################
    fname = 'pkl_files/r2center_medians_v2.pkl'
    with open(fname,'rb') as f:
        [r2center_2d_wrt_age,\
                     r2center_2d_wrt_motiv,\
                     r2center_2d_wrt_objtype] = pickle.load(f)    
        
    ftools.plot_r2center_wrt_age(r2center_medians_wrt_age)
    ftools.plot_r2center_wrt_motiv(r2center_medians_wrt_motiv)
    ftools.plot_r2center_wrt_objtype(r2center_medians_wrt_objtype)
    
    
    """
    Time elapsed  1.47 sec
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)
