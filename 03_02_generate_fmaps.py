#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:09:19 2020

@author: zeynep

This functions computes fixation maps from gaze samples (ecological, randomly 
sampled from the baseline maps or their complements).
"""
import pickle
import time
import cv2

import tools_modulation as mtools
import tools_saliency as stools

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

                      
if __name__ == "__main__":
    start_time = time.time()
    cv2.destroyAllWindows()
           
    for p, participant in enumerate( constants.PARTICIPANTS ) :
               
        print(participant)
        fmaps_eco_random_inv = mtools.init_fmaps_eco_random_inv()
            
        #  load previous computed fixations eco_random_inv
        fpath = 'pkl_files/fixations_eco_random_inv/' + participant + '.pkl'
        with open(fpath,'rb') as f:
           fixations_eco_random_inv = pickle.load(f)            
          

        fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
        
        a = person.age_range
        m = person.motiv
          
        for o in constants.OBJECT_TYPES: 
        
            for iii, (image_fname, fixations_random, fixations_eco, fixations_inv)  in enumerate(zip(\
                                              fixations_eco_random_inv[a][m][o]['image_fnames'],\
                                              fixations_eco_random_inv[a][m][o]['fixations_random'],\
                                              fixations_eco_random_inv[a][m][o]['fixations_eco'],\
                                              fixations_eco_random_inv[a][m][o]['fixations_inv'])):
            
                print(iii, end =". ") 
                
                fmap_eco, fmap_random, fmap_inv = [], [], []
                
                """
                Build fmaps eco, random, inv
                """                      
                fmap_random = stools.build_fmap_from_fixations(fixations_random, 'fmap_with_fixations_random')
                fmap_eco = stools.build_fmap_from_fixations(fixations_eco, 'fmap_with_fixations_eco')                     
                fmap_inv = stools.build_fmap_from_fixations(fixations_inv, 'fmap_with_fixations_inv')                  
                                               
                """
                Finally append both (random and eco)
                """     
                fmaps_eco_random_inv[a][m][o]['image_fnames'].append(image_fname)
                fmaps_eco_random_inv[a][m][o]['fmaps_random'].append(fmap_random)
                fmaps_eco_random_inv[a][m][o]['fmaps_eco'].append(fmap_eco)
                fmaps_eco_random_inv[a][m][o]['fmaps_inv'].append(fmap_inv)
                
            print(' ')# for new line                
            
#        output_fname = '../pkl_files/fmaps_eco_random_inv/' + participant + '.pkl'
#        with open(str(output_fname), 'wb') as f:
#            pickle.dump(fmaps_eco_random_inv, \
#                        f, pickle.HIGHEST_PROTOCOL)  
                
    """
    It takes very long to run this function (days)
    """            
        
    elapsed_time = time.time() - start_time
    print('Time elapsed for computing metrics %2.2f sec' %elapsed_time)      
        
                
