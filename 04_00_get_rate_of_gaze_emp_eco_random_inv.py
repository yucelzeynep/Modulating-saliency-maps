#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:01:45 2020

@author: zeynep

This function loads the gaze samples recorede from the participnats and counts
the number of samples in each of the following:
    functional part
    manipulative part
    neither 
    
It also reports the reulsts as a figure, arraged for ages and motivations.
"""


import numpy as np
import matplotlib.pyplot as plt

import pickle
import time
import tools_modulation as mtools

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)




    
def plot_r_fix_emp_vs_random(r_fix_stats_emp, r_fix_stats_random):
    
    plt.figure()
    

    for k in r_fix_stats_emp.keys():
        rand_col1 = (np.random.rand(),np.random.rand(),np.random.rand())
        rand_col2 = (np.random.rand(),np.random.rand(),np.random.rand())
        
        xf  = r_fix_stats_emp[k]['func']
        yf =  r_fix_stats_random[k]['func']
        
        xm  = r_fix_stats_emp[k]['manip']
        ym =  r_fix_stats_random[k]['manip']        
        
        plt.plot(xf,yf,color = rand_col1, marker= '^',label=k+'_func') 
        
        plt.plot(xm,ym, color = rand_col2, marker =  's',label=k+'_manip') 
        
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('square')
    
    
if __name__ == "__main__":
    
    start_time = time.time()
    
    r_fix_wrt_amop_emp,\
    r_fix_wrt_amop_eco,\
    r_fix_wrt_amop_random,\
    r_fix_wrt_amop_inv = \
    mtools.init_r_fix_mats(), mtools.init_r_fix_mats(), mtools.init_r_fix_mats(), mtools.init_r_fix_mats()


    for p, participant in enumerate( constants.PARTICIPANTS ) :
        
        print(participant)
        
        """
        Load empirical fixations of this person (and also age and motiv)
        """
        fpath =  constants.INPUT_DIR + 'person/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            person = pickle.load(f)
            
        a = person.age_range
        m = person.motiv
                
        """
        Load synthetic fixations
        """
        fpath = 'pkl_files/fixations_eco_random_inv/' + participant + '.pkl'
        with open(fpath,'rb') as f:
            fixations_eco_random_inv = pickle.load(f)
          
        
            
        for oi, o in enumerate(preferences.OBJECT_TYPES_INTEREST):
                        
            fixations_emp = person.image[o]['image_fixations']
            fixations_emp = mtools.remove_blank_target_fixations(fixations_emp, person.image[o]['image_fnames'] )
            
            for ijk, (fix_emp, \
                      image_fname, fix_eco, \
                      fix_random, fix_inv) in \
                      enumerate(zip(\
                                       fixations_emp,\
                                       fixations_eco_random_inv[a][m][o]['image_fnames'],\
                                       fixations_eco_random_inv[a][m][o]['fixations_eco'],\
                                       fixations_eco_random_inv[a][m][o]['fixations_random'],\
                                       fixations_eco_random_inv[a][m][o]['fixations_inv'])):
                
                
                
                if image_fname != constants.TARGET_IMAGE_FNAME and\
                    image_fname != constants.BLANK_IMAGE_FNAME:
                    
                    """
                    Load myobject foraccess the the polygons (i.e. func, manip)
                    """
                    myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                    with open(myobject_fpath,'rb') as f:
                        myobject = pickle.load(f)
                   
                                
                    r_fix_wrt_amop_emp = mtools.allocate_fixations_wrt_amop(myobject,\
                                                                    fix_emp,\
                                                                    r_fix_wrt_amop_emp,\
                                                                    a, m, o)
            
                    r_fix_wrt_amop_eco = mtools.allocate_fixations_wrt_amop(myobject,\
                                                                    fix_eco,\
                                                                    r_fix_wrt_amop_eco,\
                                                                    a, m, o)
                    
                    
                    r_fix_wrt_amop_random = mtools.allocate_fixations_wrt_amop(myobject,\
                                                                       fix_random,\
                                                                       r_fix_wrt_amop_random,\
                                                                       a, m, o)

                    r_fix_wrt_amop_inv = mtools.allocate_fixations_wrt_amop(myobject,\
                                                                    fix_inv,\
                                                                    r_fix_wrt_amop_inv,\
                                                                    a, m, o)                    

                    
    r_fix_stats_wrt_ap_emp, r_fix_stats_wrt_mp_emp, r_fix_stats_wrt_op_emp = \
    mtools.get_r_fixation_stats(r_fix_wrt_amop_emp)
    
    r_fix_stats_wrt_ap_eco, r_fix_stats_wrt_mp_eco, r_fix_stats_wrt_op_eco = \
    mtools.get_r_fixation_stats(r_fix_wrt_amop_eco)
    
    r_fix_stats_wrt_ap_random, r_fix_stats_wrt_mp_random, r_fix_stats_wrt_op_random = \
    mtools.get_r_fixation_stats(r_fix_wrt_amop_random)
    
    r_fix_stats_wrt_ap_inv, r_fix_stats_wrt_mp_inv, r_fix_stats_wrt_op_inv = \
    mtools.get_r_fixation_stats(r_fix_wrt_amop_inv)
    
    plot_r_fix_emp_vs_random(r_fix_stats_wrt_ap_emp, r_fix_stats_wrt_ap_random)
    plot_r_fix_emp_vs_random(r_fix_stats_wrt_mp_emp, r_fix_stats_wrt_mp_random)
    plot_r_fix_emp_vs_random(r_fix_stats_wrt_op_emp, r_fix_stats_wrt_op_random)
    
    # save r_fix_stats_wrt_objtype objects
    fpath =  'pkl_files/r_fix_stats_wrt_objtype_emp_eco_random_inv' + '.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([\
                      r_fix_wrt_amop_emp,\
                      r_fix_wrt_amop_eco,\
                      r_fix_wrt_amop_random,\
                      r_fix_wrt_amop_inv], \
                        f, pickle.HIGHEST_PROTOCOL)      
                                             
    """
    Time elapsed 26.54 sec
    """
              
    elapsed_time = time.time() - start_time
    print('Time elapsed %2.2f sec' %elapsed_time)      
