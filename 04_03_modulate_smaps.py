#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:54:34 2020

@author: zeynep

This function modifies the saliency map by:
    attenuating functional region (grasping)
    amplifying manipulative region
    
I load the prviously computed fixation maps (supplementary ie supp)
and also load the moficiation_coefs

and direclt take the linear combination. 

The other function (modify_smaps_with_eco_inv) 
computes fixation maps from scratch

This one is just for updating modul_coefs anf trying how the results change.
"""

import time
import pickle

import tools_saliency as stools
import tools_modulation as mtools

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)


if __name__ == '__main__':
    
    start_time = time.time()
        
    fpath = 'pkl_files/modul_coefs.pkl'
    with open(fpath,'rb') as f:
        modul_coefs = pickle.load(f)
      
    baseline_center_prior = stools.GaussianMask( \
                                     constants.IMAGE_WIDTH, \
                                     constants.IMAGE_HEIGHT, \
                                     sigma=int(constants.IMAGE_WIDTH/3))
    
    metrics_emp2orig, metrics_emp2modif_eco,\
       metrics_emp2modif_random = stools.init_metrics()
    
    for p, participant in enumerate( constants.PARTICIPANTS ) :
        
        print(participant)        
       
        fpath =  constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
        a = person.age_range
        m = person.motiv            
            
        """
        The following (supp_maps_all) is the set of supp_map_eco_pos, supp_map_neg and supp_map_modif
        It includes a linear combination of 
        smap and supmaps with some stupid coefs (1.1 for amplify and 0.9 for attenuate)
        
        After I comouted modif_coef based on counting gaze pts in polygons
        I decided to change these 1.1 and 0.9
        so here i dont compute supp_maps from scract
        but only take the linear combination with these new coefs
        """
        fpath = 'pkl_files/modified_smaps_and_supp/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            supp_maps_all = pickle.load(f)     
            
                   
        for o in preferences.OBJECT_TYPES_INTEREST:
            for (image_fname, supp_map_eco_pos, supp_map_random_pos, supp_map_neg) \
             in zip(supp_maps_all[a][m][o]['image_fnames'],\
                    supp_maps_all[a][m][o]['supp_map_eco_pos'],\
                    supp_maps_all[a][m][o]['supp_map_random_pos'],\
                    supp_maps_all[a][m][o]['supp_map_neg']):
                
                if image_fname != constants.TARGET_IMAGE_FNAME and\
                    image_fname != constants.BLANK_IMAGE_FNAME:
                        
                    # load myobject
                    myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                        
                    ind = person.image[o]['image_fnames'].index(image_fname)
                    fmap_observed = person.image[o]['fmaps'][ind ]
                                   
               
                    smap_orig = myobject.smap['STATIC_SPECRES']
                    smap_orig_norm = stools.normalize_map(smap_orig) 
                    
                    smap_eco_modif_f = mtools.superimpose_maps_f(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)
                    
                    smap_eco_modif_m = mtools.superimpose_maps_m(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)

                    smap_eco_modif_fm = mtools.superimpose_maps_fm(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)   
                        
                    smap_random_modif_f = mtools.superimpose_maps_f(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)
                    
                    smap_random_modif_m = mtools.superimpose_maps_m(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)

                    smap_random_modif_fm = mtools.superimpose_maps_fm(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modul_coefs,
                                                  a,m,o)   
                    
                    fmap_observed_norm = stools.normalize_map(fmap_observed) 
                    
                    smap_eco_modif_f_norm = stools.normalize_map(smap_eco_modif_f)
                    smap_eco_modif_m_norm = stools.normalize_map(smap_eco_modif_m)
                    smap_eco_modif_fm_norm = stools.normalize_map(smap_eco_modif_fm)
                    
                    smap_random_modif_f_norm = stools.normalize_map(smap_random_modif_f)
                    smap_random_modif_m_norm = stools.normalize_map(smap_random_modif_m)
                    smap_random_modif_fm_norm = stools.normalize_map(smap_random_modif_fm)
                    
                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2orig[a][m][o][sm].append( stools.compute_metric(sm, smap_orig_norm, fmap_observed_norm, baseline_center_prior) )
                   
                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2modif_eco[a][m][o][sm]['f'].append( stools.compute_metric(sm, smap_eco_modif_f_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_eco[a][m][o][sm]['m'].append( stools.compute_metric(sm, smap_eco_modif_m_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_eco[a][m][o][sm]['fm'].append( stools.compute_metric(sm, smap_eco_modif_fm_norm, fmap_observed_norm, baseline_center_prior) )

                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2modif_random[a][m][o][sm]['f'].append( stools.compute_metric(sm, smap_random_modif_f_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_random[a][m][o][sm]['m'].append( stools.compute_metric(sm, smap_random_modif_m_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_random[a][m][o][sm]['fm'].append( stools.compute_metric(sm, smap_random_modif_fm_norm, fmap_observed_norm, baseline_center_prior) )

                    
    # save values of saliency metrics
    fpath = 'pkl_files/metrics_v4.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([metrics_emp2orig,\
                     metrics_emp2modif_eco,\
                     metrics_emp2modif_random], f, pickle.HIGHEST_PROTOCOL)      
         
    """
    Time elapsed 3491.93 sec    
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

