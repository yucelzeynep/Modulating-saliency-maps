#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:12:55 2020

@author: zeynep

This function gives KOPT_MAX_CLUSTERS cluster center estimates for each image.

In doing that, employs the standard saliency map (with BASELINE_SALIENCY_TYPE, see
preference for details) and densely samples this smap, i.e. returning many 
random points drawn from the samp.

It then clusters these points into KOPT_MAX_CLUSTERS clusters. 

The resulting cluster center estimates are evaluated on the smap and their 
potential saliency is found. The estimates are then sorted in decreasing order 
of saliency (i.e. most salienct estimate at the begining of the array) and 
pickled.
"""

import numpy as np
import pickle
import time

from sklearn.cluster import KMeans

import tools_reaction as rtools

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)
        
if __name__ == "__main__":
    
    start_time = time.time()
    
    participant = constants.PARTICIPANTS[0] 
    fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
    with open(fpath,'rb') as f:
        temp_person = pickle.load(f)
    
    cluster_center_candidates = {}

    for object_type in preferences.OBJECT_TYPES_INTEREST:
        
        cluster_center_candidates[object_type] = {}
        
        for image_fname in  temp_person.image[object_type]['image_fnames']:
            
            if image_fname != constants.TARGET_IMAGE_FNAME and\
                image_fname != constants.BLANK_IMAGE_FNAME:

                myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                with open(myobject_fpath,'rb') as f:
                    myobject = pickle.load(f)
               
        
                random_saccades = rtools.sample_from_smap(myobject.image_orig, \
                                                       myobject.smap[preferences.BASELINE_SALIENCY_TYPE])
                
                fixations_unq = rtools.get_fixations_unq(random_saccades)
                
                k_means_cluster_centers, k_means_labels, _ = rtools.cluster_by_kmeans(fixations_unq, preferences.KOPT_MAX_CLUSTERS)
                
                k_means_cluster_centers_sorted = rtools.sort_cluster_centers_wrt_saliency(\
                                                                                   myobject.smap[preferences.BASELINE_SALIENCY_TYPE], \
                                                                                   k_means_cluster_centers)
    

                
                cluster_center_candidates[object_type][image_fname] = k_means_cluster_centers_sorted
                
                
    # save cluster_center_candidates dict object
    fpath = 'pkl_files/cluster_center_candidates.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump(cluster_center_candidates, \
                    f, pickle.HIGHEST_PROTOCOL)                 


    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time) # about 4 sec

