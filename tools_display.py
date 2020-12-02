#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:11:28 2020

@author: zeynep
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

def scale_and_display(image_orig, window_title):
    """
    This function receives an image (possibly with some drawings on),
    scales it and displays
    """
    # for scaling
    if len(image_orig.shape) == 2:
        H, W = image_orig.shape
    else:
        H, W, _ = image_orig.shape
        
    scaled_height = int( H * preferences.SCALE_PERCENT ) 
    scaled_width = int( W  * preferences.SCALE_PERCENT )
    DIM = (scaled_width, scaled_height)
    
    image_temp = cv2.resize(image_orig, DIM, interpolation = cv2.INTER_AREA) 
    cv2.imshow(window_title, image_temp)
    

def display_image_with_clusters(myobject, saccades, \
                                fixations_unq, k_means_cluster_centers, k_means_labels):
    """
    This function display gaze samples over an object image. Each cluster is 
    displayed in a different color.
    """
    
    # random colors
    cluster_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255))
             for i in range(preferences.KMAX)]
    
    
    image_temp = myobject.image_orig.copy()
    gray3c = cv2.merge([image_temp, image_temp, image_temp])
    

        
    # clustered points (fixations_unq) in random colors
    for k, col in zip(range(len(k_means_cluster_centers)), cluster_colors):
        cluster_pts_inds = np.where( k_means_labels == k )[0]
    
        
        cluster_center = k_means_cluster_centers[k]
        for mm in cluster_pts_inds:
            
            cv2.circle(gray3c,\
                       (int(fixations_unq[mm][0]),  int(fixations_unq[mm][1])),
                       10, col, -1)
        
        cv2.circle(gray3c,\
                   (int(cluster_center[0]), int(cluster_center[1])), \
                   15, col, -1)
        
    scale_and_display(gray3c, 'image with gaze samples')
    
    cv2.waitKey(2)

def plot_displacement_hist(ddd, varname):
    """
    This functions plots empirical pdf of displacement. Note that it does not 
    receive a pdf. It takes the  raw data points and computes the histogram and 
    scales it to a pfd. It also saves the values of the pdf in a txt file. I use 
    those in gnuplot.
    """
    plt.figure()
    for k in ddd.keys():
        ddd_log  = np.log( ddd[k] ) 
            
        hist, bin_edges = np.histogram(ddd_log, bins = preferences.NBINS_LOG)   
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= k.split('_')[-1][0:3]  )
        

        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        # Here we transpose our data, so that we have it in two columns        
        datafile_path = 'figures/emp_displacement_between_fixations_'  + k.split('_')[-1][0:3] + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
                        
            
    plt.grid(linestyle='--', linewidth=1)    
    plt.legend()
    plt.show()
    

def plot_nsaccade_hists(nsaccades_wrt_age,\
                         nsaccades_wrt_motiv,\
                         nsaccades_wrt_objtype):
    """
    This function plots the pdf of k-optimum (kopt).
    
    kopt is the optimum k for k-means clustering computed according to Silhouette
    method. I also export the pdf values into a txt file, for using them later 
    in gnuplot.
    """
        
    ########################################################
    plt.figure()   
    for age_range in constants.AGE_RANGES:
         
        hist, bin_edges = np.histogram(nsaccades_wrt_age[age_range], \
                                       bins=10, range = (0,10))
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= age_range)
    
        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nsaccades_wrt_age_'  + age_range + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('Kopt')
    plt.ylabel('N views')
    plt.show()

        
    ########################################################
    plt.figure() 
    for m in constants.MOTIVATIONS:
           
        hist, bin_edges = np.histogram(nsaccades_wrt_motiv[m], \
                                       bins=10, range = (0,10))
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= m )
    
        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nsaccades_wrt_motiv_'  + m + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('Kopt')
    plt.ylabel('N views')
    plt.show()
   
        

    ########################################################
    plt.figure()    
    for o in preferences.OBJECT_TYPES_INTEREST:
        hist, bin_edges = np.histogram(nsaccades_wrt_objtype[o], \
                                       bins=10, range = (0,10))
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= o.split('_')[-1][0:3] )
    
        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nsaccades_wrt_objtype_'  + o.split('_')[-1][0:3] + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('Kopt')
    plt.ylabel('N views')
    plt.show()
    

    
def plot_nfixations_per_cluster_hists(nfixations_per_cluster_wrt_age,\
                         nfixations_per_cluster_wrt_motiv,\
                         nfixations_per_cluster_wrt_objtype):
    """
    Plot nfixations hists
    """
        
    ########################################################
    plt.figure()   
    for age_range in constants.AGE_RANGES:
         
        hist, bin_edges = np.histogram(nfixations_per_cluster_wrt_age[age_range])
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= age_range)
    

        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nfixations_per_cluster_wrt_age_'  + age_range + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('n_saccades')
    plt.ylabel('Number of instances')
    plt.show()
    

        
    ########################################################
    plt.figure() 
    for m in constants.MOTIVATIONS:
           
        hist, bin_edges = np.histogram(nfixations_per_cluster_wrt_motiv[m])
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= m)
    
        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nfixations_per_cluster_wrt_motiv_'  + m + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('n_saccades')
    plt.ylabel('Number of instances')
    plt.show()

    ########################################################
    plt.figure()    
    for o in preferences.OBJECT_TYPES_INTEREST:
        hist, bin_edges = np.histogram(nfixations_per_cluster_wrt_objtype[o])
        
        bin_size = bin_edges[1] - bin_edges[0]
        hist  = hist / np.sum(hist) / bin_size
        
        plt.plot(bin_edges[1:], hist,\
                                       label= o.split('_')[-1][0:3] )
    
        data = np.array([bin_edges[1:],\
                         hist\
                         ])
        data = data.T
        datafile_path = 'figures/nfixations_per_cluster_wrt_objtype_'  + o.split('_')[-1][0:3] + '.txt'
        with open(datafile_path, 'w+') as datafile_id:        
            np.savetxt(datafile_id, data, fmt=['%f','%f'])
        
   
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)    
    plt.xlabel('n_saccades')
    plt.ylabel('Number of instances')
    plt.show()
    
        

def plot_emp_and_synth_pdfs(pdf_synth, pdf_emp, bin_edges_emp):
    
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:    
            plt.figure()
            plt.title(age_range + '_' +object_type )
            for mi, m in enumerate( constants.MOTIVATIONS ):    
                p = plt.plot(bin_edges_emp[age_range][m][object_type][0:-1], pdf_synth[age_range][m][object_type], '-', label=m+'_synth')
                plt.plot(bin_edges_emp[age_range][m][object_type][0:-1], pdf_emp[age_range][m][object_type], '--', \
                         color=p[0].get_color(), label=m+'_emp')
        
            plt.grid(linestyle='--', linewidth=1)    
            plt.legend()
            plt.show()
