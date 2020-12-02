#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:19:35 2020

@author: zeynep
"""

"""
This function modifies the salinecy map by:
    attenuating functional region (grasping)
    amplifying manipulative region
    
For 1, I use the original saliency map, S_orig, and randomly sample some points. 
I build a dense fixation map, S_att. 

for 2, I use the inverse saliency map, and ramply sample points. I build a dense
fixation map, S_amp.

I add S_att and S_amp to compute the S_modif. (modification)

I then take a linear combination of the  S_orig and S_modif to compute the 
adjusted map.


"""

import cv2
import time

import pickle

import tools_modulation as mtools

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)



if __name__ == '__main__':
    
    start_time = time.time()
    
    cv2.destroyAllWindows()
    
    smaps = []
    
    for p, participant in enumerate( constants.PARTICIPANTS ) :
        
        print(participant)
        
        smaps_supp_modif = mtools.init_supp_maps_eco_random_inv()
        
        fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
        a = person.age_range
        m = person.motiv
            
        fpath = 'pkl_files/fixations_eco_random_inv/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            fixations_eco_random_inv = pickle.load(f)
          
        for oi, o in enumerate(preferences.OBJECT_TYPES_INTEREST):
            
            
            for iii, (image_fname, \
                fixations_eco, \
                fixations_random, \
                fixations_inv) in enumerate(zip(\
                                                   fixations_eco_random_inv[a][m][o]['image_fnames'],\
                                                   fixations_eco_random_inv[a][m][o]['fixations_eco'],\
                                                   fixations_eco_random_inv[a][m][o]['fixations_random'],\
                                                   fixations_eco_random_inv[a][m][o]['fixations_inv'])):
                
                print(iii, end =". ") 

                if image_fname != constants.TARGET_IMAGE_FNAME and\
                    image_fname != constants.BLANK_IMAGE_FNAME:
                    
                    # load myobject
                    myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                     
                    smap = myobject.smap['STATIC_SPECRES']
                    
                    poly_func = myobject.polygon_functional # handle, grip
                    poly_manip = myobject.polygon_manipulative # end effector                   
                
                    fixations_eco_masked = mtools.mask_fixations_with_poly(fixations_eco, poly_manip)
                    fixations_random_masked = mtools.mask_fixations_with_poly(fixations_random, poly_manip)
                    fixations_inv_masked = mtools.mask_fixations_with_poly(fixations_inv, poly_func)
                        
                    supp_map_eco_pos, supp_map_random_pos, supp_map_neg = \
                        mtools.get_supp_maps(fixations_eco_masked, \
                                      fixations_random_masked,\
                                      fixations_inv_masked)
    
                    # smap_eco_modif = superimpose_maps(smap, \
                    #                               supp_map_eco_pos, \
                    #                               supp_map_neg)
                
                    # smap_random_modif = superimpose_maps(smap, \
                    #                               supp_map_random_pos, \
                    #                               supp_map_neg)                    
       
 
                    
                    smaps_supp_modif[a][m][o]['image_fnames'].append(image_fname)
                    
                    smaps_supp_modif[a][m][o]['supp_map_eco_pos'].append(supp_map_eco_pos)
                    smaps_supp_modif[a][m][o]['supp_map_random_pos'].append(supp_map_random_pos)

                    smaps_supp_modif[a][m][o]['supp_map_neg'].append(supp_map_neg)
                    
                    # # these two are never used, right?
                    # smaps_supp_modif[a][m][o]['smap_eco_modif'].append(smap_eco_modif)
                    # smaps_supp_modif[a][m][o]['smap_random_modif'].append(smap_random_modif)

                    
            print(' ') # just for new line            
                    
        # save smaps_supp_modif object
        fpath = 'pkl_files/modified_smaps_and_supp/' + participant + '.pkl'
        with open(str(fpath), 'wb') as f:
            pickle.dump(smaps_supp_modif, f, pickle.HIGHEST_PROTOCOL)            

    """
    Time elapsed  35424.13 sec
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

