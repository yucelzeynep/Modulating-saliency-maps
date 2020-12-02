#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:04:48 2020

@author: zeynep

In this file, wesort the values of empirical observations of the variables (e.g. 
nscaades, displacements etc.) into arrays. Some of those arrays distinguish only
 a single (intrinsic or extrinsic) feature and the others care for every possible 
 combination (of age, motiv, objtype). I then compute the empirical pdfs from those.

"""

import matplotlib.pyplot as plt

import tools_reaction as rtools

import pickle
import time
 
from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

    
if __name__ == "__main__":
    
    start_time = time.time()
    plt.close()
    
    d_wrt_age, d_wrt_motiv, d_wrt_objtype, d_wrt_age_motiv_objtype = \
    rtools.init_displacement_between_fixations()
    
    dx_wrt_age, dx_wrt_motiv, dx_wrt_objtype, dx_wrt_age_motiv_objtype = \
    rtools.init_displacement_between_fixations()
    
    dy_wrt_age, dy_wrt_motiv, dy_wrt_objtype, dy_wrt_age_motiv_objtype = \
    rtools.init_displacement_between_fixations()
       
    
    if preferences.FROM_SCRATCH:
        """
        if FROM_SCRATCH key is True, 
        do all the calculations from start
        """
        nsaccades_wrt_age, nsaccades_wrt_motiv, nsaccades_wrt_objtype, nsaccades_wrt_age_motiv_objtype, \
        nfixations_per_cluster_wrt_age, nfixations_per_cluster_wrt_motiv, \
        nfixations_per_cluster_wrt_objtype, nfixations_per_cluster_wrt_age_motiv_objtype =\
        rtools.init_nclusters_nfixations()
            
        for p, participant in enumerate( constants.PARTICIPANTS ) :
                        
            print(participant)
            
            fpath = constants.INPUT_DIR + 'person/' + participant + '.pkl'
            with open(fpath,'rb') as f:
                person = pickle.load(f)
              
            for object_type in preferences.OBJECT_TYPES_INTEREST:
                for image_fname, saccades in zip(\
                                                       person.image[object_type]['image_fnames'],\
                                                       person.image[object_type]['image_fixations']):
                    if image_fname != constants.TARGET_IMAGE_FNAME and\
                        image_fname != constants.BLANK_IMAGE_FNAME:
                            
                            
                            # load myobject
                            myobject_fpath = constants.INPUT_DIR + 'images/' + image_fname.replace('jpeg', 'pkl')
                            with open(myobject_fpath,'rb') as f:
                                myobject = pickle.load(f)
                            
                            """
                            Find optimum number of clusters by silhouette method
                            and use that to do the clustering
                            
                            Append the optimum number to an array
                            """
                            sil, kopt, fixations_unq = rtools.find_Kopt_silhouette(saccades)
                            k_means_cluster_centers, k_means_labels, nfixations_per_clustertemp = rtools.cluster_by_kmeans(fixations_unq, kopt)
                            
                            nsaccades_wrt_age[person.age_range].append(kopt)
                            nsaccades_wrt_motiv[person.motiv].append(kopt)
                            nsaccades_wrt_objtype[object_type].append(kopt)
                            nsaccades_wrt_age_motiv_objtype[person.age_range][person.motiv][object_type].append(kopt)

                            nfixations_per_cluster_wrt_age[person.age_range].extend(nfixations_per_clustertemp)
                            nfixations_per_cluster_wrt_motiv[person.motiv].extend(nfixations_per_clustertemp)
                            nfixations_per_cluster_wrt_objtype[object_type].extend(nfixations_per_clustertemp)
                            nfixations_per_cluster_wrt_age_motiv_objtype[person.age_range][person.motiv][object_type].append(nfixations_per_clustertemp)
                            
                            

                            
                            """
                            Within each cluster, get the displacement between 
                            consecutive gaze samples
                            """
                            d, dx, dy = rtools.get_displacement_between_fixations_within_cluster(fixations_unq,\
                                                                        k_means_cluster_centers, \
                                                                        k_means_labels)
                            d_wrt_age[person.age_range].extend( d ) 
                            d_wrt_motiv[person.motiv].extend( d )                             
                            d_wrt_objtype[object_type].extend( d ) 
                            d_wrt_age_motiv_objtype[person.age_range][person.motiv][object_type].extend( d )
                            
                            dx_wrt_age[person.age_range].extend( dx ) 
                            dx_wrt_motiv[person.motiv].extend( dx )                             
                            dx_wrt_objtype[object_type].extend( dx )       
                            dx_wrt_age_motiv_objtype[person.age_range][person.motiv][object_type].extend( dx )
                            
                            dy_wrt_age[person.age_range].extend( dy ) 
                            dy_wrt_motiv[person.motiv].extend( dy )                             
                            dy_wrt_objtype[object_type].extend( dy )    
                            dy_wrt_age_motiv_objtype[person.age_range][person.motiv][object_type].extend( dy )
                            
                            
                            if preferences.DISPLAY_PROCESSING:
                                rtools.display_image_with_clusters(myobject, saccades, \
                                                            fixations_unq, \
                                                            k_means_cluster_centers, \
                                                            k_means_labels)
                      
                

            
        # save kopt list object
        fpath = 'pkl_files/nsaccades_wrt_age_motiv_objtype.pkl'
        with open(str(fpath), 'wb') as f:
            pickle.dump([\
                         nsaccades_wrt_age,\
                         nsaccades_wrt_motiv,\
                         nsaccades_wrt_objtype,\
                         nsaccades_wrt_age_motiv_objtype], \
                            f, pickle.HIGHEST_PROTOCOL)              
            
        # save n_saccades_per_fixation list object
        fpath = 'pkl_files/nfixations_per_cluster_wrt_age_motiv_objtype.pkl'
        with open(str(fpath), 'wb') as f:
            pickle.dump([\
                        nfixations_per_cluster_wrt_age,\
                        nfixations_per_cluster_wrt_motiv,\
                        nfixations_per_cluster_wrt_objtype,\
                        nfixations_per_cluster_wrt_age_motiv_objtype], \
                            f, pickle.HIGHEST_PROTOCOL)               
            
        # save displacement_between_fixations_wrt_age_motiv_objtype object
        fpath = 'pkl_files/displacement_between_fixations_wrt_age_motiv_objtype.pkl'
        with open(str(fpath), 'wb') as f:
            pickle.dump([d_wrt_age,\
                         d_wrt_motiv,\
                         d_wrt_objtype,\
                         d_wrt_age_motiv_objtype,\
                         dx_wrt_age,\
                         dx_wrt_motiv,\
                         dx_wrt_objtype,\
                         dx_wrt_age_motiv_objtype,\
                         dy_wrt_age,\
                         dy_wrt_motiv,\
                         dy_wrt_objtype,\
                         dy_wrt_age_motiv_objtype], \
                            f, pickle.HIGHEST_PROTOCOL)  
                        
    else:
        """
        if FROM_SCRATCH key is False, 
        load the previoulsy calculated data
        """      
        
        print('FROM_SCRATCH is False. Loading old data!')
        
        fpath = 'pkl_files/nsaccades_wrt_age_motiv_objtype.pkl'
        with open(fpath,'rb') as f:
            [\
                         nsaccades_wrt_age,\
                         nsaccades_wrt_motiv,\
                         nsaccades_wrt_objtype,\
                         nsaccades_wrt_age_motiv_objtype] = pickle.load(f)
            
        
        fpath = 'pkl_files/nfixations_per_cluster_wrt_age_motiv_objtype.pkl'
        with open(fpath,'rb') as f:
            [\
                        nfixations_per_cluster_wrt_age,\
                        nfixations_per_cluster_wrt_motiv,\
                        nfixations_per_cluster_wrt_objtype,\
                        nfixations_per_cluster_wrt_age_motiv_objtype] = pickle.load(f)
            
        fpath = 'pkl_files/displacement_between_fixations_wrt_age_motiv_objtype.pkl'
        with open(fpath,'rb') as f:
            [d_wrt_age,\
             d_wrt_motiv,\
             d_wrt_objtype,\
             d_wrt_age_motiv_objtype,\
             dx_wrt_age,\
             dx_wrt_motiv,\
             dx_wrt_objtype,\
             dx_wrt_age_motiv_objtype,\
             dy_wrt_age,\
             dy_wrt_motiv,\
             dy_wrt_objtype,\
             dy_wrt_age_motiv_objtype]= pickle.load(f)                 
            
    rtools.plot_nsaccade_hists(nsaccades_wrt_age,\
                         nsaccades_wrt_motiv,\
                         nsaccades_wrt_objtype)
    
    rtools.plot_nfixations_per_cluster_hists(\
                                     nfixations_per_cluster_wrt_age,\
                                     nfixations_per_cluster_wrt_motiv,\
                                     nfixations_per_cluster_wrt_objtype)
    

    """
    Since preprocess_displacement is adapted to three-stage structure 
    (i.e. age-motiv-objtype), this part requires a modified version of 
    preprocessing. Plotting function is ok but currently it is redundant 
    without the preprocessing
    """
    
    d_wrt_age = rtools.preprocess_displacement_wrt_amo(d_wrt_age)
    d_wrt_motiv = rtools.preprocess_displacement_wrt_amo(d_wrt_motiv)
    d_wrt_objtype = rtools.preprocess_displacement_wrt_amo(d_wrt_objtype)
    
    rtools.plot_displacement_hist(d_wrt_age, 'age')
    rtools.plot_displacement_hist(d_wrt_motiv, 'motiv')
    rtools.plot_displacement_hist(d_wrt_objtype, 'objtype')
   
                                  
    """
    Time elapsed for computing metrics 243.24 sec
    """                         
    elapsed_time = time.time() - start_time
    print('Time elapsed for computing metrics %2.2f sec' %elapsed_time)      