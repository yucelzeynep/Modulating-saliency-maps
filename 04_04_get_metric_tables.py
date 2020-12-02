#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:54:34 2020

@author: zeynep

This function returns the improvement in represing the actual gaze behavior with 
a saliency map, in terms of a set of saliency metrics. To that end, we find it
 meaning to give the relative (i.e. percentage) values of improvement than their 
absolute values. Namely, the metrics are computed in very different ways and a
direct comparison between them is not possible. 
"""

import numpy as np
import time

import pickle

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

def get_perc_imp(f2m, f2o):
    """
    This function returns the improvement in terms of a given saliency metric 
    as a percentage. Here, f2m is the valie of the metric with respect to a 
    modulated saliency map and f2o is the value of the metric with respect to the
    original (i.e. baseline) salinecy map. Since the metrics are computed in very 
    different ways, we found it more meaning to give the relative (i.e. percentage)
    improvement than absolute value. 
    """
    return (np.mean(f2m) - np.mean(f2o)/ np.abs(np.mean(f2o)))*100

    
if __name__ == '__main__':
    
    start_time = time.time()
        
    fpath = 'pkl_files/metrics_v3.pkl'
    with open(fpath,'rb') as f:
        [metrics_emp2orig,\
         metrics_emp2modif_eco,\
         metrics_emp2modif_random] = pickle.load(f)
        
    tables_eco_a_vs_sm_wrt_func_manip = {}
    tables_random_a_vs_sm_wrt_func_manip = {}
    for sm in preferences.SALIENCY_METRICS:
        tables_eco_a_vs_sm_wrt_func_manip[sm] = {}
        tables_random_a_vs_sm_wrt_func_manip[sm] = {}
        for a in constants.AGE_RANGES:
            tables_eco_a_vs_sm_wrt_func_manip[sm][a] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}
            tables_random_a_vs_sm_wrt_func_manip[sm][a] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}
    tables_eco_m_vs_sm_wrt_func_manip = {}
    tables_random_m_vs_sm_wrt_func_manip = {}
    for sm in preferences.SALIENCY_METRICS:
        tables_eco_m_vs_sm_wrt_func_manip[sm] = {}
        tables_random_m_vs_sm_wrt_func_manip[sm] = {}
        for m in constants.MOTIVATIONS:
            tables_eco_m_vs_sm_wrt_func_manip[sm][m] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}
            tables_random_m_vs_sm_wrt_func_manip[sm][m] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}
    
    tables_eco_o_vs_sm_wrt_func_manip = {}
    tables_random_o_vs_sm_wrt_func_manip = {}
    for sm in preferences.SALIENCY_METRICS:
        tables_eco_o_vs_sm_wrt_func_manip[sm] = {}
        tables_random_o_vs_sm_wrt_func_manip[sm] = {}
        for o in constants.OBJECT_TYPES:
            tables_eco_o_vs_sm_wrt_func_manip[sm][o] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}
            tables_random_o_vs_sm_wrt_func_manip[sm][o] = {\
                                           'f':[],\
                                           'm':[],\
                                           'fm':[]}                
                
                
    for a in constants.AGE_RANGES:
        for m in constants.MOTIVATIONS:
            for o in constants.OBJECT_TYPES:
                for sm in preferences.SALIENCY_METRICS:
                    
                    tables_eco_a_vs_sm_wrt_func_manip[sm][a]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_a_vs_sm_wrt_func_manip[sm][a]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_a_vs_sm_wrt_func_manip[sm][a]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))
                    
                    tables_eco_m_vs_sm_wrt_func_manip[sm][m]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_m_vs_sm_wrt_func_manip[sm][m]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_m_vs_sm_wrt_func_manip[sm][m]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))                    
                    
                    tables_eco_o_vs_sm_wrt_func_manip[sm][o]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_o_vs_sm_wrt_func_manip[sm][o]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_eco_o_vs_sm_wrt_func_manip[sm][o]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_eco[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))                        
                    
                        
                    tables_random_a_vs_sm_wrt_func_manip[sm][a]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_a_vs_sm_wrt_func_manip[sm][a]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_a_vs_sm_wrt_func_manip[sm][a]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))
                    
                    tables_random_m_vs_sm_wrt_func_manip[sm][m]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_m_vs_sm_wrt_func_manip[sm][m]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_m_vs_sm_wrt_func_manip[sm][m]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))                    
                    
                    tables_random_o_vs_sm_wrt_func_manip[sm][o]['f'].append( \
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['f'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_o_vs_sm_wrt_func_manip[sm][o]['m'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['m'], metrics_emp2orig[a][m][o][sm]))
                    tables_random_o_vs_sm_wrt_func_manip[sm][o]['fm'].append(\
                                                 get_perc_imp(metrics_emp2modif_random[a][m][o][sm]['fm'], metrics_emp2orig[a][m][o][sm]))                        
                    ###############################################################################  
    print('------------ECOLOGICAL--------')                    
    print('----------------------------------')                    
    print('Modifying only func')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_eco_a_vs_sm_wrt_func_manip[sm][a]['f'])), end='\t')
        print(' ')
      
    print('\n\nMotiv')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_eco_m_vs_sm_wrt_func_manip[sm][m]['f'])), end='\t')
        print(' ')
        
    print('\n\nOBJTYPE')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_eco_o_vs_sm_wrt_func_manip[sm][o]['f'])), end='\t')
        print(' ')
###############################################################################                    
    print('----------------------------------')                    
    print('Modifying only manip')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_eco_a_vs_sm_wrt_func_manip[sm][a]['m'])), end='\t')
        print(' ')
        
    print('\n\nMotiv')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_eco_m_vs_sm_wrt_func_manip[sm][m]['m'])), end='\t')
        print(' ')
        
    print('\n\nOBJTYPE')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_eco_o_vs_sm_wrt_func_manip[sm][o]['m'])), end='\t')
        print(' ')
        
###############################################################################                    
    print('----------------------------------')                    
    print('Modifying both func and  manip')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_eco_a_vs_sm_wrt_func_manip[sm][a]['fm'])), end='\t')
        print(' ')
        
    print('\n\nMotiv')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_eco_m_vs_sm_wrt_func_manip[sm][m]['fm'])), end='\t')
        print(' ')
            
    print('\n\nOBJTYPE')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_eco_o_vs_sm_wrt_func_manip[sm][o]['fm'])), end='\t')
        print(' ')
        
        
        
        
        
        
        
        
        

    print('------------RANDOM--------')                    
    print('----------------------------------')                    
    print('Modifying only func')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_random_a_vs_sm_wrt_func_manip[sm][a]['f'])), end='\t')
        print(' ')
      
    print('\n\nMotiv')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_random_m_vs_sm_wrt_func_manip[sm][m]['f'])), end='\t')
        print(' ')
        
    print('\n\nOBJTYPE')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_random_o_vs_sm_wrt_func_manip[sm][o]['f'])), end='\t')
        print(' ')
###############################################################################                    
    print('----------------------------------')                    
    print('Modifying only manip')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_random_a_vs_sm_wrt_func_manip[sm][a]['m'])), end='\t')
        print(' ')
        
    print('\n\nMotiv')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_random_m_vs_sm_wrt_func_manip[sm][m]['m'])), end='\t')
        print(' ')
        
    print('\n\nOBJTYPE')
      
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_random_o_vs_sm_wrt_func_manip[sm][o]['m'])), end='\t')
        print(' ')
        
###############################################################################                    
    print('----------------------------------')                    
    print('Modifying both func and  manip')
    print('\nAge')
    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for a in constants.AGE_RANGES:
            print('{0:0.5f}'.format(np.mean(tables_random_a_vs_sm_wrt_func_manip[sm][a]['fm'])), end='\t')
        print(' ')
        
    print('\n\nMotiv')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for m in constants.MOTIVATIONS:
            print('{0:0.5f}'.format(np.mean(tables_random_m_vs_sm_wrt_func_manip[sm][m]['fm'])), end='\t')
        print(' ')
            
    print('\n\nOBJTYPE')

    for sm in preferences.SALIENCY_METRICS:
        print(sm,end='\t')
        for o in constants.OBJECT_TYPES:
            print('{0:0.5f}'.format(np.mean(tables_random_o_vs_sm_wrt_func_manip[sm][o]['fm'])), end='\t')
        print(' ')     
        
        
        
        
        
        
    """
    Time elapsed 0.1 sec    
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)

