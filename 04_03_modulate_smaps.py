#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:54:34 2020

@author: zeynep
"""


"""
This function modifies the saliency map by:
    attenuating functional region (grasping)
    amplifying manipulative region
    
I load the prviously computed fixation maps (supplementary ie supp)
and also load the moficiation_coefs

and direclt take the linear combination. 

The other function (modify_smaps_with_eco_inv) 
computes fixation maps from scratch

This one is just for updating modif_coefs anf trying how the results change.
"""

import numpy as np
import time

import pickle

import tools_saliency as stools

import sys
sys.path.insert(0, '../')# for file_tools etc
sys.path.insert(0, '../arrange_exp_data/')#

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)



def init_smaps_supp_modif():
    """
    Init smaps supp and modif
    
    One for each participant
    """            
    smaps_supp_modif = {}
    
    for object_type in np.sort( preferences.OBJECT_TYPES_INTEREST):
                   
        smaps_supp_modif[object_type] = {\
                                'image_fnames': [],\
                                'supp_map_eco_pos': [],\
                                'supp_map_neg': [],\
                                'smap_modif_f':[],\
                                'smap_modif_m':[],\
                                'smap_modif_fm':[]   }
    
    return smaps_supp_modif

def init_metrics():
    
    metrics_emp2orig = {}
    metrics_emp2modif_eco = {}
    metrics_emp2modif_random = {}
    
    for a in constants.AGE_RANGES:
        metrics_emp2orig[a], metrics_emp2modif_eco[a], metrics_emp2modif_random[a] = \
            {}, {}, {}
        for m in constants.MOTIVATIONS:
            metrics_emp2orig[a][m],metrics_emp2modif_eco[a][m], \
                metrics_emp2modif_random[a][m] = {}, {}, {}
            for o in constants.OBJECT_TYPES:
                metrics_emp2orig[a][m][o],metrics_emp2modif_eco[a][m][o],\
                    metrics_emp2modif_random[a][m][o] = {}, {}, {}
                for sm in preferences.SALIENCY_METRICS:
                    metrics_emp2orig[a][m][o][sm] = []
                    metrics_emp2modif_eco[a][m][o][sm] = {'f':[], 'm':[],'fm':[]}
                    metrics_emp2modif_random[a][m][o][sm] = {'f':[], 'm':[],'fm':[]}

    return metrics_emp2orig, metrics_emp2modif_eco, metrics_emp2modif_random
                    


def superimpose_maps_f(smap_orig, \
                       supp_map_pos, \
                       supp_map_neg,\
                       modif_coefs, \
                       a, m, o):
    """
    Scale up manipulative part
    scale down functional part
    
    Actually up or down is decided by sign of modif_coef 
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
#    mask_pos = (supp_map_pos>0) * 255    
#    mask_pos = mask_pos.astype(np.uint8)    
#    
#     # this is usually positive so i call it amplify
#    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
#    
#    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
#    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
#    smap_modif[smap_modif> 255] = 255
                        
    mask_neg = (supp_map_neg>0) * 255    
    mask_neg = mask_neg.astype(np.uint8)    
    
    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
    
    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
    smap_modif[smap_modif> 255] = 255
   
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif

def superimpose_maps_m(smap_orig, \
                       supp_map_pos, \
                       supp_map_neg, \
                       modif_coefs, \
                       a, m, o):
    """
    Scale up manipulative part
    scale down functional part
    
    Actually up or down is decided by sign of modif_coef 
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
    mask_pos = (supp_map_pos>0) * 255    
    mask_pos = mask_pos.astype(np.uint8)    
    
     # this is usually positive so i call it amplify
    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
    
    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
    smap_modif[smap_modif> 255] = 255
                        
#    mask_neg = (supp_map_neg>0) * 255    
#    mask_neg = mask_neg.astype(np.uint8)    
#    
#    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
#    
#    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
#    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
#    smap_modif[smap_modif> 255] = 255
   
  
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif
    

def superimpose_maps_fm(smap_orig, \
                        supp_map_pos, \
                        supp_map_neg, \
                        modif_coefs, \
                        a,m,o):
    """
    Scale up manipulative part
    scale down functional part
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
    mask_pos = (supp_map_pos>0) * 255    
    mask_pos = mask_pos.astype(np.uint8)    
    
     # this is usually positive so i call it amplify
    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
    
    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
    smap_modif[smap_modif> 255] = 255
                        
    mask_neg = (supp_map_neg>0) * 255    
    mask_neg = mask_neg.astype(np.uint8)    
    
    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
    
    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
    smap_modif[smap_modif> 255] = 255
   
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif

def compute_metric(saliency_metric, smap, fmap, baseline_center_prior):
 
    if saliency_metric is 'AUC_JUDD':
        return stools.auc_judd(smap, fmap)
    elif saliency_metric is  'AUC_BORJI':
        return stools.auc_borji(smap, fmap)
    elif saliency_metric is  'AUC_SHUFF':
        return stools.auc_shuff(smap, fmap)
    elif saliency_metric is  'NSS':
        return stools.nss(smap, fmap)
    elif saliency_metric is  'INFOGAIN':
        return stools.infogain(smap, fmap, baseline_center_prior)
    elif saliency_metric is  'SIM':
        return stools.similarity(smap, fmap)
    elif saliency_metric is  'CC':
        return stools.cc(smap, fmap)
    elif saliency_metric is  'KLDIV':
        return stools.kldiv(smap, fmap)
    else:
        print('No such metric as ', saliency_metric)
        return 0
        
    
    


    
if __name__ == '__main__':
    
    start_time = time.time()
        
    fpath = 'pkl_files/modif_coefs.pkl'
    with open(fpath,'rb') as f:
        modif_coefs = pickle.load(f)
      
    baseline_center_prior = stools.GaussianMask( \
                                     constants.IMAGE_WIDTH, \
                                     constants.IMAGE_HEIGHT, \
                                     sigma=int(constants.IMAGE_WIDTH/3))
    
    metrics_emp2orig, metrics_emp2modif_eco,\
       metrics_emp2modif_random = init_metrics()
    
    for p, participant in enumerate( constants.PARTICIPANTS ) :
        
        print(participant)        
       
        
        
        fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
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
                    myobject_fpath =  constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                        
                    ind = person.image[o]['image_fnames'].index(image_fname)
                    fmap_observed = person.image[o]['fmaps'][ind ]
                                   
               
                    smap_orig = myobject.smap['STATIC_SPECRES']
                    smap_orig_norm = stools.normalize_map(smap_orig) 
                    
                    smap_eco_modif_f = superimpose_maps_f(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)
                    
                    smap_eco_modif_m = superimpose_maps_m(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)

                    smap_eco_modif_fm = superimpose_maps_fm(smap_orig, \
                                                  supp_map_eco_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)   
                        
                    smap_random_modif_f = superimpose_maps_f(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)
                    
                    smap_random_modif_m = superimpose_maps_m(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)

                    smap_random_modif_fm = superimpose_maps_fm(smap_orig, \
                                                  supp_map_random_pos, \
                                                  supp_map_neg,\
                                                  modif_coefs,
                                                  a,m,o)   
                    
                    fmap_observed_norm = stools.normalize_map(fmap_observed) 
                    
                    smap_eco_modif_f_norm = stools.normalize_map(smap_eco_modif_f)
                    smap_eco_modif_m_norm = stools.normalize_map(smap_eco_modif_m)
                    smap_eco_modif_fm_norm = stools.normalize_map(smap_eco_modif_fm)
                    
                    smap_random_modif_f_norm = stools.normalize_map(smap_random_modif_f)
                    smap_random_modif_m_norm = stools.normalize_map(smap_random_modif_m)
                    smap_random_modif_fm_norm = stools.normalize_map(smap_random_modif_fm)
                    
                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2orig[a][m][o][sm].append( compute_metric(sm, smap_orig_norm, fmap_observed_norm, baseline_center_prior) )
                  
                    
                    
                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2modif_eco[a][m][o][sm]['f'].append( compute_metric(sm, smap_eco_modif_f_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_eco[a][m][o][sm]['m'].append( compute_metric(sm, smap_eco_modif_m_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_eco[a][m][o][sm]['fm'].append( compute_metric(sm, smap_eco_modif_fm_norm, fmap_observed_norm, baseline_center_prior) )

                    
               
                    for sm in preferences.SALIENCY_METRICS:
                        metrics_emp2modif_random[a][m][o][sm]['f'].append( compute_metric(sm, smap_random_modif_f_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_random[a][m][o][sm]['m'].append( compute_metric(sm, smap_random_modif_m_norm, fmap_observed_norm, baseline_center_prior) )
                        metrics_emp2modif_random[a][m][o][sm]['fm'].append( compute_metric(sm, smap_random_modif_fm_norm, fmap_observed_norm, baseline_center_prior) )

                    
    # save values of saliency metrics
    fpath = 'pkl_files/metrics_v3.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([metrics_emp2orig,\
                     metrics_emp2modif_eco,\
                     metrics_emp2modif_random], f, pickle.HIGHEST_PROTOCOL)      
         
    """
    Time elapsed 2260.93 sec without judd
    Time elapsed  27641.15 sec with judd
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

