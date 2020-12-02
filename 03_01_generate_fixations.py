#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:09:19 2020

@author: zeynep

This file contains generates three kinds of -synthetic- fixations:
    1. Ecological fixations: these are the ones that try to replicate the 
    gazing pattern of human subjects, i.e. including the saccadic movements 
    2. Randomly sampled from smap: These are some random samples which are drawn 
    from a salinecy map, assuming that it describes a 2D pdf. 
    3. Randomly sampled from the complement of smap: These are very similar to 
    the above, but that they consider the underlying pdf to be complement of the 
    previous one (used in 2.). 
"""

import pickle
import time

import cv2

import tools_file as ftools
import tools_modulation as mtools

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

if __name__ == "__main__":
    
    start_time = time.time()
    
    cv2.destroyAllWindows()
               
    cdf_emp_nsaccades, \
    bin_edges_emp_nsaccades,\
    cdf_emp_nfixations, \
    bin_edges_emp_nfixations,\
    cdf_d_wrt_age_motiv_objtype, \
    cdf_dx_wrt_age_motiv_objtype, \
    cdf_dy_wrt_age_motiv_objtype,\
    bin_edges_emp_d_wrt_age_motiv_objtype,\
    bin_edges_emp_dx_wrt_age_motiv_objtype,\
    bin_edges_emp_dy_wrt_age_motiv_objtype = ftools.load_cdf_emp()
    
    cluster_center_candidates = ftools.load_cluster_center_candidates()
        
    for p, participant in enumerate( constants.PARTICIPANTS ) :
       
#        if p<10:
#            continue
        
        print(participant)
        fixations_eco_random_inv = mtools.init_fixations_eco_random_inv()

        
        fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
        
        a = person.age_range
        m = person.motiv

        for o in preferences.OBJECT_TYPES_INTEREST: 
            
            for iii, image_fname in enumerate( person.image[o]['image_fnames'] ):
                
                print(iii, end =". ") 
                
                fmap_eco, fixations_eco, fmap_random, fixations_random = [], [], [], []
                
                if image_fname != constants.TARGET_IMAGE_FNAME and\
                image_fname != constants.BLANK_IMAGE_FNAME:
                    
                    myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')                  
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                  
                    
                    """
                    Saliency map and its complement:
                        smap, smap_inv
                    """
                    smap = myobject.smap[preferences.BASELINE_SALIENCY_TYPE]
                    smap_inv = (255-smap)
                    
                    """
                    Get three kinds of fixations:
                         1. Ecological 
                         2. Randomly sampled from smap
                         3. Randomly sampled from the complement of smap
                    """
                    
                    fixations_eco = mtools.get_fixations_eco(myobject, cluster_center_candidates, image_fname,\
                                                      a, m, o,\
                                                      cdf_emp_nsaccades, \
                                                      bin_edges_emp_nsaccades,\
                                                      cdf_emp_nfixations, \
                                                      bin_edges_emp_nfixations,\
                                                      cdf_d_wrt_age_motiv_objtype, \
                                                      cdf_dx_wrt_age_motiv_objtype, \
                                                      cdf_dy_wrt_age_motiv_objtype,\
                                                      bin_edges_emp_d_wrt_age_motiv_objtype,\
                                                      bin_edges_emp_dx_wrt_age_motiv_objtype,\
                                                      bin_edges_emp_dy_wrt_age_motiv_objtype)
                        
                        
                    fixations_random = \
                    mtools.get_fixations_ri(myobject.image_orig, smap)

                    
                    fixations_inv = \
                    mtools.get_fixations_ri(myobject.image_orig, smap_inv)
                    
                                                 
                    """
                    Finally, append the fixations to the output arrays
                    """     
                    fixations_eco_random_inv[a][m][o]['image_fnames'].append(image_fname)
                    fixations_eco_random_inv[a][m][o]['fixations_random'].append(fixations_random)                    
                    fixations_eco_random_inv[a][m][o]['fixations_eco'].append(fixations_eco)
                    fixations_eco_random_inv[a][m][o]['fixations_inv'].append(fixations_inv)                    
                                       

                
            print(' ') # just for a new line                
            
#        output_fname = 'pkl_files/fixations_eco_random_inv/' + participant + '.pkl'
#        with open(str(output_fname), 'wb') as f:
#            pickle.dump(fixations_eco_random_inv, \
#                        f, pickle.HIGHEST_PROTOCOL)  
                
        
    """
    This function takes about 3 hours to run
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed %2.2f sec' %elapsed_time)      
        
                