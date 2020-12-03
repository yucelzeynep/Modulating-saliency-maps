#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:20:19 2020

@author: zeynep
"""
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)


def plot_r2center_overall(r2center_stats_overall):
    
    fr, axr = plt.subplots()
    axr.set_xlim([0,100])
    axr.set_ylim([0,400])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for age_range in constants.AGE_RANGES:
            
        p = plt.plot(r2center_stats_overall[age_range]['means'], label=age_range+'_means')
        plt.plot(r2center_stats_overall[age_range]['medians'],  '--', color=p[0].get_color(), label=age_range+'_medians')
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('N pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
    
    plt.legend()
    plt.show()
    
            
def plot_r2center_wrt_object_types(r2center_medians_wrt_object_types):
    
    fr, axr = plt.subplots()
    axr.set_xlim([0,100])
    axr.set_ylim([0,400])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40, 100+int(constants.IMAGE_HEIGHT * preferences.SCALE_PERCENT), 400, 400)
    
    for o in constants.OBJECT_TYPES:
        p = plt.plot(r2center_medians_wrt_object_types['means'][o], label=o)
        plt.plot(r2center_medians_wrt_object_types['medians'][o], '--', color=p[0].get_color())
        plt.xlabel('time (0-1 sec)')
        plt.ylabel('n pixels to target center')
        plt.grid(linestyle='--', linewidth=1)
        
    plt.legend()
    plt.show()
    

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
        
        # Here we transpose your data, so to as have it in two columns    
        # This txt file will be used in gnuplot later
        data = data.T
        
        datafile_path = constants.OUTPUT_DIR_FIG  + \
        'r2center_medians_wrt_age_' + \
        age_range + '.txt'
        
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%d','%d'])
    
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
        # Here we transpose your data, so to as have it in two columns    
        # This txt file will be used in gnuplot later
        data = data.T
        
        datafile_path = constants.OUTPUT_DIR_FIG  + \
        'r2center_medians_wrt_motiv_'  + \
        m + '.txt'
        
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%d','%d'])
        
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
    
    for o in constants.OBJECT_TYPES:
        
        plt.plot(r2center_medians_wrt_objtype[o], label=o.split('_')[-1][0:3])

        #######################################################################
        # save wrt objtype
        #
        data = np.array([range(len(r2center_medians_wrt_objtype[o])),\
                         r2center_medians_wrt_objtype[o]\
                         ])
        # Here we transpose your data, so to as have it in two columns    
        # This txt file will be used in gnuplot later
        data = data.T
        
        datafile_path = constants.OUTPUT_DIR_FIG   + \
        'r2center_medians_wrt_objtype_'  + \
        o.split('_')[-1][0:3] + '.txt'
        
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%d','%d'])

    plt.xlabel('time (0-1 sec)')
    plt.ylabel('n pixels to target center')
    plt.grid(linestyle='--', linewidth=1)
        
    plt.legend()
    plt.show()
    

        