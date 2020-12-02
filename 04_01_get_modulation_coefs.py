#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:30:33 2020

@author: zeynep
"""
import numpy as np
import time 
 
from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

import pickle

if __name__ == "__main__":
    
    start_time = time.time()
        
    fpath =  'pkl_files/r_fix_stats_wrt_objtype_emp_eco_random_inv' + '.pkl'
    with open(fpath,'rb') as f:
            [\
                      r_fix_wrt_amop_emp,\
                      r_fix_wrt_amop_eco,\
                      r_fix_wrt_amop_random,\
                      r_fix_wrt_amop_inv] = pickle.load(f)
                
    modif_coefs = {}
    for a in constants.AGE_RANGES:
        modif_coefs[a] = {} 
        for m in constants.MOTIVATIONS:
            modif_coefs[a][m] = {} 
            for o in np.sort( preferences.OBJECT_TYPES_INTEREST):
                modif_coefs[a][m][o] = {} 
                for p in constants.OBJECT_PARTS: 
                    modif_coefs[a][m][o][p] = 0
                    
                    modif_coefs[a][m][o][p] =  \
                    (np.mean(r_fix_wrt_amop_emp[a][m][o][p]) -\
                     np.mean(r_fix_wrt_amop_random[a][m][o][p])) / \
                     np.mean(r_fix_wrt_amop_random[a][m][o][p])
                     
    # save modif_coefs
    fpath = 'pkl_files/modif_coefs.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump(modif_coefs, f, pickle.HIGHEST_PROTOCOL)                    
        
    """
    Time elapsed for computing metrics 0.01 sec
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed for computing metrics %2.2f sec' %elapsed_time)      