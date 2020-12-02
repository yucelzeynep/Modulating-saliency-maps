#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:52:06 2020

@author: zeynep

This file contains the fucntions in managing file operations. 
"""
import os
import fnmatch

import pickle

import constants
from importlib import reload
reload(constants)


def clear_all_files_under_dir(output_dir):
    """
    Clear all files in the output_dir recursively 
    (only files, not directories)
    """
        
    for root, dirs, files in os.walk(output_dir):    
        for name in files:
            print(os.path.join(root, name)) 
            os.remove(os.path.join(root, name))
        


def find_file_in_path(pattern, path):
    """
    This function finds a file (i.e. file name), that includes a certain pattern 
    e.g. *_gaze.txt under a given path
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result   


def load_cdf_emp():
    """
    Load empirical cdfs and corresponding bin_edges
    """
    
    fpath = 'pkl_files/cdf_emp_nsaccades.pkl'
    with open(fpath,'rb') as f:
       [cdf_emp_nsaccades, \
        bin_edges_emp_nsaccades] = pickle.load(f)
       
       
    fpath = 'pkl_files/cdf_emp_nfixations.pkl'
    with open(fpath,'rb') as f:
        [cdf_emp_nfixations, \
         bin_edges_emp_nfixations] = pickle.load(f)
        

    fpath = 'pkl_files/cdf_emp_displacement_btw_fixations.pkl'
    with open(fpath,'rb') as f:
       [cdf_d_wrt_age_motiv_objtype, \
        cdf_dx_wrt_age_motiv_objtype, \
        cdf_dy_wrt_age_motiv_objtype,\
        bin_edges_emp_d_wrt_age_motiv_objtype,\
        bin_edges_emp_dx_wrt_age_motiv_objtype,\
        bin_edges_emp_dy_wrt_age_motiv_objtype] = pickle.load(f)
       
    return cdf_emp_nsaccades, \
        bin_edges_emp_nsaccades,\
        cdf_emp_nfixations, \
        bin_edges_emp_nfixations,\
        cdf_d_wrt_age_motiv_objtype, \
        cdf_dx_wrt_age_motiv_objtype, \
        cdf_dy_wrt_age_motiv_objtype,\
        bin_edges_emp_d_wrt_age_motiv_objtype,\
        bin_edges_emp_dx_wrt_age_motiv_objtype,\
        bin_edges_emp_dy_wrt_age_motiv_objtype
        

    
def load_cluster_center_candidates():
    """
    These are based on saliency map.
    So they are inherent properties of the images and they do not change for 
    varying participants.
    """
    
    fpath = 'pkl_files/cluster_center_candidates.pkl'
    with open(fpath,'rb') as f:
       cluster_center_candidates = pickle.load(f)
    
    return cluster_center_candidates