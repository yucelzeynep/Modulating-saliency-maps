#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:12:55 2020

@author: zeynep

This function gives KOPT cluster center estimates for each image.

In doing that, employs the standard saliency map (with SALIENCY_TYPE) and
densely samples this smap, i.e. returning many random points drawn from the samp.

It then clusters these points into KOPT clusters. 

The resulting cluster center estimates are evaluated on the smap and their 
potential saliency is found. The estimates are then sorted in decreasing order 
of saliency (i.e. most salienct estimate at the begining of the array) and 
pickled.
"""

import numpy as np

from sklearn.cluster import KMeans

import pickle
import time

import sys

sys.path.insert(0, '../arrange_exp_data')# for person class
from person import Person

sys.path.insert(0, '../') # for constants, and preferences

from importlib import reload
import preferences
reload(preferences)
import constants
reload(constants)

SALIENCY_TYPE = 'STATIC_SPECRES'
N_OVERSAMPLE = 1000
KOPT = 10 # cluster into 10
    
    
def removeDuplicates(lst): 
    """
    from a list, remove duplicate instances of values (leave a unique instance)
    """
    return [t for t in (set(tuple(i) for i in lst))] 

def get_saccades_unq(saccades):
    """
    Get unique saccades
    
    The dense sampling returns many points (artificial gaze samples). It is 
    unlikely that any two points are at the same location. But just to be
    sure, I run this little scipt to remove duplicates.
    """
    temp  = []
    for s in saccades:
        temp.append([s[0], s[1]])
    
    temp = removeDuplicates(temp)
    saccades_unq = tuple(temp)
    
    return saccades_unq
    
def cluster_by_kmeans(saccades_unq, kopt):
    """
    In this file, I do **not** estimate kopt (with elbow method or silh.)
    
    Namely, I assume a fixed number for optimal number of cluster.
    
    This assures that I am not left out without a cluster center estimate in 
    the actual routine.
    
    n_init is the number of times the k-means algorithm will be run with different 
    centroid seeds. The final results will be the best output of n_init 
    consecutive runs in terms of inertia. (default=10)

    """
    k_means = KMeans(init='k-means++', n_clusters=kopt, n_init=10)
    k_means.fit(saccades_unq)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    
    return k_means_cluster_centers, k_means_labels
    
def sample_pts_from_smap(image_orig, smap):
    """
    This function considers smap as a 2d pdf. It randomly samples points from 
    this distribution. These can be considered as articial saccades or gaze 
    points. 
    
    The sampling is dense, i.e. I sample too many points. See above for 
    N_OVERSAMPLE. 
    
    Many of the estimated cluster centers will not be used eventually, but I do 
    not want to have a run time error when I search a value at a (large) index 
    of the cluster_centers array.
    """
    
    if len(image_orig.shape) == 2:
        H, W = image_orig.shape
    else:
        H, W, _ = image_orig.shape
            
    smap_normalized = smap / smap.sum() # it has to be normalized 
    
    
    # Make n random selections from the flattened pmf without replacement
    inds = np.random.choice(\
                            np.arange(W*H),\
                            p=smap_normalized.reshape(-1), \
                            size=N_OVERSAMPLE, replace=False)
    
    temp_random_saccades = np.unravel_index(inds, (H,W))
    
    random_saccades = []
    for (x,y) in zip(temp_random_saccades[0], temp_random_saccades[1]):
        random_saccades.append([y,x])
       
    return random_saccades


def sort_cluster_centers_wrt_saliency(smap, k_means_cluster_centers):
    """
    After I get the cluster center estimates, I check their saliency in the 
    original saliency map.
    
    I then sort the estimates from the one with largest saliency, towards the 
    one with smallest saliency.
    
    When I generate artificial fixation patterns, I start from the begining of 
    this array (my first artificial fixation most likely to be attended).
    """
    
    saliencies_at_cluster_centers = []
    
    for cc in k_means_cluster_centers:
        saliencies_at_cluster_centers.append(
                smap[int(cc[1])][ int(cc[0])])
        
    # to do it in descending order I needed a trick
    # argsort sorts in ascending order
    # so I flip the indices array
    inds = np.array( saliencies_at_cluster_centers ).argsort()[::-1] 
    
    k_means_cluster_centers_sorted = []
    for iii in inds:
        k_means_cluster_centers_sorted.append( k_means_cluster_centers[iii] )
        
    return k_means_cluster_centers_sorted
    
        
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
               
        
                random_saccades = sample_pts_from_smap(myobject.image_orig, \
                                                       myobject.smap[SALIENCY_TYPE])
                
                saccades_unq = get_saccades_unq(random_saccades)
                
                k_means_cluster_centers, k_means_labels = cluster_by_kmeans(saccades_unq, KOPT)
                
                k_means_cluster_centers_sorted = sort_cluster_centers_wrt_saliency(\
                                                                                   myobject.smap[SALIENCY_TYPE], \
                                                                                   k_means_cluster_centers)
    

                
                cluster_center_candidates[object_type][image_fname] = k_means_cluster_centers_sorted
                
                
#    # save cluster_center_candidates dict object
#    fpath = 'cluster_center_candidates.pkl'
#    with open(str(fpath), 'wb') as f:
#        pickle.dump(cluster_center_candidates, \
#                    f, pickle.HIGHEST_PROTOCOL)                 


    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time) # about 4 sec

