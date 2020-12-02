#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:31:28 2020

@author: zeynep
"""
import matplotlib.pyplot as plt
import time
import pickle

import tools_reaction_to_object_image as rtools

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

    
if __name__ == "__main__":
    
    start_time = time.time()
    plt.close()     
    
        
    """
    Load entire raw arrays
    
    """
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
    """
    For building empirical pdf,
    first get histograms and then scale them properly
    """
    
    pdf_emp_nsaccades, bin_edges_emp_nsaccades = rtools.get_pdfs_emp(nsaccades_wrt_age_motiv_objtype, myrange = (0,10), mybins= 0)
    cdf_emp_nsaccades = rtools.get_cdf_emp(pdf_emp_nsaccades)
    
    pdf_emp_nfixations, bin_edges_emp_nfixations = rtools.get_pdfs_emp(nfixations_per_cluster_wrt_age_motiv_objtype, 0, 0)
    cdf_emp_nfixations = rtools.get_cdf_emp(pdf_emp_nfixations)
    
    d_wrt_age_motiv_objtype, dx_wrt_age_motiv_objtype, dy_wrt_age_motiv_objtype = \
    rtools.preprocess_displacement(d_wrt_age_motiv_objtype), \
    rtools.preprocess_displacement(dx_wrt_age_motiv_objtype), \
    rtools.preprocess_displacement(dy_wrt_age_motiv_objtype)
    
    d_log_wrt_age_motiv_objtype, \
    dx_log_wrt_age_motiv_objtype, \
    dy_log_wrt_age_motiv_objtype = \
    rtools.get_log_displacement(d_wrt_age_motiv_objtype), \
    rtools.get_log_displacement(dx_wrt_age_motiv_objtype),\
    rtools.get_log_displacement(dy_wrt_age_motiv_objtype)
    
    pdf_emp_d_log_wrt_age_motiv_objtype, bin_edges_emp_d_wrt_age_motiv_objtype = \
    rtools.get_pdfs_emp(d_log_wrt_age_motiv_objtype, myrange =0, mybins = preferences.NBINS_LOG)
    
    pdf_emp_dx_log_wrt_age_motiv_objtype, bin_edges_emp_dx_wrt_age_motiv_objtype = \
    rtools.get_pdfs_emp(dx_log_wrt_age_motiv_objtype, myrange =0, mybins = preferences.NBINS_LOG)
    
    pdf_emp_dy_log_wrt_age_motiv_objtype, bin_edges_emp_dy_wrt_age_motiv_objtype = \
    rtools.get_pdfs_emp(dy_log_wrt_age_motiv_objtype, myrange =0, mybins = preferences.NBINS_LOG)    
    
    (cdf_d_wrt_age_motiv_objtype, cdf_dx_wrt_age_motiv_objtype, cdf_dy_wrt_age_motiv_objtype) = \
    rtools.get_cdf_emp(pdf_emp_d_log_wrt_age_motiv_objtype), \
    rtools.get_cdf_emp(pdf_emp_dx_log_wrt_age_motiv_objtype), \
    rtools.get_cdf_emp(pdf_emp_dy_log_wrt_age_motiv_objtype)
        
    # save cdf objects
    fpath = 'pkl_files/cdf_emp_nsaccades.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([cdf_emp_nsaccades, \
                     bin_edges_emp_nsaccades],\
                    f, pickle.HIGHEST_PROTOCOL)   
            
    fpath = 'pkl_files/cdf_emp_nfixations.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([cdf_emp_nfixations, \
                     bin_edges_emp_nfixations],\
                    f, pickle.HIGHEST_PROTOCOL)   
            
    fpath = 'pkl_files/cdf_emp_displacement_btw_fixations.pkl'
    with open(str(fpath), 'wb') as f:
        pickle.dump([cdf_d_wrt_age_motiv_objtype, \
                     cdf_dx_wrt_age_motiv_objtype, \
                     cdf_dy_wrt_age_motiv_objtype,\
                     bin_edges_emp_d_wrt_age_motiv_objtype,\
                     bin_edges_emp_dx_wrt_age_motiv_objtype,\
                     bin_edges_emp_dy_wrt_age_motiv_objtype],\
                     f, pickle.HIGHEST_PROTOCOL)  
    
    
    """
    Sample random points (n_clusters)
    """
    random_samples_nsaccades = rtools.sample_from_emp_distribution(cdf_emp_nsaccades, bin_edges_emp_nsaccades, preferences.N_RANDOM_SAMPLES)
    random_samples_nfixations = rtools.sample_from_emp_distribution(cdf_emp_nfixations, bin_edges_emp_nfixations, preferences.N_RANDOM_SAMPLES)


    random_samples_d_log_wrt_age_motiv_objtype, random_samples_dx_log_wrt_age_motiv_objtype, random_samples_dy_log_wrt_age_motiv_objtype = rtools.sample_from_emp_distribution_log_displacement(\
                                 cdf_d_wrt_age_motiv_objtype, bin_edges_emp_d_wrt_age_motiv_objtype,\
                                 cdf_dx_wrt_age_motiv_objtype, bin_edges_emp_dx_wrt_age_motiv_objtype,\
                                 cdf_dy_wrt_age_motiv_objtype, bin_edges_emp_dy_wrt_age_motiv_objtype,\
                                 preferences.N_RANDOM_SAMPLES)

    """
    Building the pdfs from random samples
    """
    pdf_synth_nsaccades = rtools.get_synth_pdf(random_samples_nsaccades, bin_edges_emp_nsaccades)
    pdf_synth_nfixations = rtools.get_synth_pdf(random_samples_nfixations, bin_edges_emp_nfixations)
    pdf_synth_d_log_wrt_age_motiv_objtype = \
    rtools.get_synth_pdf_displacement(random_samples_dx_log_wrt_age_motiv_objtype, \
                  random_samples_dy_log_wrt_age_motiv_objtype, \
                  bin_edges_emp_d_wrt_age_motiv_objtype)
                          

    rtools.plot_emp_and_synth_pdfs(pdf_synth_nsaccades, pdf_emp_nsaccades, bin_edges_emp_nsaccades)
    rtools.plot_emp_and_synth_pdfs(pdf_synth_nfixations, pdf_emp_nfixations, bin_edges_emp_nfixations )
    rtools.plot_emp_and_synth_pdfs(pdf_synth_d_log_wrt_age_motiv_objtype, \
                                   pdf_emp_d_log_wrt_age_motiv_objtype, \
                                   bin_edges_emp_d_wrt_age_motiv_objtype)

                                 
    """
    Time elapsed 1.10 sec
    """
    elapsed_time = time.time() - start_time
    print('Time elapsed %2.2f sec' %elapsed_time)       
