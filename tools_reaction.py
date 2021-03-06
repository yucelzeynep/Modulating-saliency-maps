#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:14:31 2020

@author: zeynep
"""

import numpy as np
import random

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)

def init_displacement_between_fixations():
    """
    This function initializes one array for each intrinsic or extrinsic feature. 
    In addition, it initializes another array for every combination of those.
    """
    d_wrt_age = {}
    for age_range in constants.AGE_RANGES:
        d_wrt_age[age_range] = []
        
    d_wrt_motiv = {}        
    for m in constants.MOTIVATIONS:
        d_wrt_motiv[m] = []
        
    d_wrt_objtype = {}
    for objtype in preferences.OBJECT_TYPES_INTEREST:
        d_wrt_objtype[objtype] = []\
        
    d_wrt_age_motiv_objtype = {}
    for age_range in constants.AGE_RANGES:        
        d_wrt_age_motiv_objtype[age_range]  = {} 
        for m in constants.MOTIVATIONS:            
            d_wrt_age_motiv_objtype[age_range][m] = {}      
            for object_type in np.sort( constants.OBJECT_TYPES):
                d_wrt_age_motiv_objtype[age_range][m][object_type] = []
            
    return d_wrt_age,\
        d_wrt_motiv, \
        d_wrt_objtype,\
        d_wrt_age_motiv_objtype


def init_nclusters_nfixations():
    """
    This function initializes number of clusters (nclusters) and number of 
    fixations per cluster (nfixations) wrt each age, motivation, object type.
    
    The shape is outarray[a][m][o],  which is the same as those for other 
    variables. 
            
    """        
    nsaccades_wrt_age, \
    nsaccades_wrt_motiv, \
    nsaccades_wrt_objtype = {}, {}, {}
    
    for age_range in constants.AGE_RANGES:        
        nsaccades_wrt_age[age_range] = []
    
    for m in constants.MOTIVATIONS:            
        nsaccades_wrt_motiv[m] = []
        
    for object_type in np.sort( constants.OBJECT_TYPES):
        nsaccades_wrt_objtype[object_type] = []
        
        
    nfixations_per_cluster_wrt_age, \
    nfixations_per_cluster_wrt_motiv, \
    nfixations_per_cluster_wrt_objtype,\
    nfixations_per_cluster_wrt_age_motiv_objtype = {}, {}, {}, {}
    
    for age_range in constants.AGE_RANGES:        
        nfixations_per_cluster_wrt_age[age_range] = []
    
    for m in constants.MOTIVATIONS:            
        nfixations_per_cluster_wrt_motiv[m] = []
        
    for object_type in np.sort( constants.OBJECT_TYPES):
        nfixations_per_cluster_wrt_objtype[object_type] = []
        
    nsaccades_wrt_age_motiv_objtype, nfixations_per_cluster_wrt_age_motiv_objtype = {}, {}
    for age_range in constants.AGE_RANGES:        
        nsaccades_wrt_age_motiv_objtype[age_range], nfixations_per_cluster_wrt_age_motiv_objtype[age_range]  = {}, {}        
        for m in constants.MOTIVATIONS:            
            nsaccades_wrt_age_motiv_objtype[age_range][m], nfixations_per_cluster_wrt_age_motiv_objtype[age_range][m] = {}, {}       
            for object_type in np.sort( constants.OBJECT_TYPES):
                nsaccades_wrt_age_motiv_objtype[age_range][m][object_type], \
                nfixations_per_cluster_wrt_age_motiv_objtype[age_range][m][object_type]= [], []
                
                
        
    return  nsaccades_wrt_age, nsaccades_wrt_motiv, nsaccades_wrt_objtype, nsaccades_wrt_age_motiv_objtype, \
    nfixations_per_cluster_wrt_age, nfixations_per_cluster_wrt_motiv, \
    nfixations_per_cluster_wrt_objtype, nfixations_per_cluster_wrt_age_motiv_objtype
   
    
def init_pdf_emp():
    """
    This function initializes empirical pdf (either nfixations or nsaccades), 
    pdf_emp, contrasting age, motivations, object_type.
    
    The shape is as follows:
        pdf_emp[age_range][m][object_type] 
            
    """        
    pdf_emp = {}
    for age_range in constants.AGE_RANGES:  
        pdf_emp[age_range] = {}      
        for m in constants.MOTIVATIONS:            
            pdf_emp[age_range][m] = {}     
            for object_type in np.sort( constants.OBJECT_TYPES):
                pdf_emp[age_range][m][object_type] = []
                

    return pdf_emp
       
    
def removeDuplicates(lst): 
    """
    From a list, remove duplicate instances of values (leave one unique instance)
    """
    return [t for t in (set(tuple(i) for i in lst))]     

    
def get_fixations_unq(fixations):
    """
    This function get unique fixations from oversampled saliency maps.
    
    The dense sampling returns many points (synthetic gaze samples). It is 
    unlikely that any two points are at the same location. But just to be
    sure, I run this little scipt to remove duplicates.
    """
    temp  = []
    for s in fixations:
        temp.append([s[0], s[1]])
    
    temp = removeDuplicates(temp)
    fixations_unq = tuple(temp)
    
    return fixations_unq
        
def preprocess_displacement_small(dist):
    """
    Remove inf, -inf, and zeros in the displacement array, which is necessary 
    to take the log transformation. 
    """    
   
    # remove inf                
    dist2 = np.asarray(dist)
    if np.sum(np.isinf(dist)) == 0:
        dist2 = dist
    else:
        replace_inf = np.isinf(dist2)  
        dist2 = dist
        dist2[replace_inf] = np.max(dist2)
    
    # remove -inf                
    mynegdist= [ -x for x in dist2]
    if np.sum(np.isinf(mynegdist)) == 0:
        dist3 = dist2
    else:
        replace_minf = np.isinf(mynegdist)
        dist3 = dist2
        dist3[replace_minf] = np.min(dist2)
    
    # remove nan                
    if np.sum(np.isnan(dist3)) == 0:
        dist4 = dist3
    else:
        dist4 = dist3
        replace_nan = np.isnan(dist4)
        dist4[replace_nan] = np.nanmean(dist3)
    
    # remove zeros                
    #print('Number of zeros {}'.format(len(dist4)-len ( np.array(dist4).nonzero()[0] )))
    dist5 = list(np.array(dist4)[np.array(dist4).nonzero()])
    
    return dist5

    
def preprocess_displacement(d_wrt_age_motiv_objtype):
    """
    This function runs the preprocessing subroutine on all arrays corresponding 
    to different combinations of intrinsic (i.e. age, motiv) and extrinsic 
    (i.e. objtype) features.
    """
    
    for a in (constants.AGE_RANGES):
        for m in constants.MOTIVATIONS:   
            for o in ( constants.OBJECT_TYPES ):
                d_wrt_age_motiv_objtype[a][m][o]  = preprocess_displacement_small( d_wrt_age_motiv_objtype[a][m][o] )
       
    return d_wrt_age_motiv_objtype


def preprocess_displacement_wrt_amo(d_wrt_amo):
    """
    This function runs the preprocessing subroutine on all arrays arranged 
    according to a single feature (for instace, grouping based on only age and 
    paying no regard to motiv or objtype). This kind of array is used only for 
    illustration purposes. See Fig. 4 of the manuscript.
    """
    for k in d_wrt_amo.keys():
        d_wrt_amo[k]  = preprocess_displacement_small( d_wrt_amo[k] )

    return d_wrt_amo

def get_log_displacement(d_wrt_age_motiv_objtype):
    
    d_log_wrt_age_motiv_objtype = {}
    
    for a in (constants.AGE_RANGES):
        d_log_wrt_age_motiv_objtype[a] = {}
        for m in constants.MOTIVATIONS:      
            d_log_wrt_age_motiv_objtype[a][m] = {}
            for o in ( constants.OBJECT_TYPES ):
                d_log_wrt_age_motiv_objtype[a][m][o] = np.log( np.abs( d_wrt_age_motiv_objtype[a][m][o])   )
    
    return  d_log_wrt_age_motiv_objtype    


def get_displacement_between_fixations_within_cluster( fixations_unq, \
                                                      k_means_cluster_centers, \
                                                      k_means_labels):
    """
    This function computes the displacement from one fixation to the next within
    each cluster,  and builds a displacement array.
    
    I compute also the displacement along x- and y-axes but I never use it.
    """       
    d, dx, dy = [], [], []
    
    # clustered points (fixations_unq) in random colors
    for k in range(len(k_means_cluster_centers)):
        cluster_pts_inds = np.where( k_means_labels == k )[0]
        
        cluster_pts_x = [fixations_unq[ind][0] for ind in cluster_pts_inds]
        cluster_pts_y = [fixations_unq[ind][1] for ind in cluster_pts_inds]
        
        dx_temp = np.diff(  np.round(cluster_pts_x) )
        dy_temp = np.diff(  np.round(cluster_pts_y) )
        d_temp = ( np.sqrt( np.add(np.square(dx_temp), \
                              np.square(dy_temp))))
        

        d.extend(d_temp)
        dx.extend(dx_temp)
        dy.extend(dy_temp)
        
    return d, dx, dy



def find_Kopt_silhouette(saccades):
    """
    This function finds optimum K for K-means clustering using the Silhouette 
    method.
    
    From 
    https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    """
    temp  = []
    for s in saccades:
        temp.append([s[0], s[1]])
    
    temp = removeDuplicates(temp)
    fixations_unq = tuple(temp)
    
    sil = []
    
    # Dissimilarity would not be defined for a single cluster, 
    # Thus, minimum number of clusters should be 2
    if len(fixations_unq) > 1:
        for k in range(2, preferences.KMAX+1):
          kmeans = KMeans(n_clusters = k).fit( fixations_unq )
          labels = kmeans.labels_
          sil.append(silhouette_score( fixations_unq , labels, metric = 'euclidean'))
          
        kopt = np.argmax(sil) + 2
        
    else:
        print('Too few unique saccades: {}. Kopt = 1'.format(len(fixations_unq)))
        kopt = 1
    
    return sil, kopt, fixations_unq


def cluster_by_kmeans(fixations_unq, kopt):
    """
    n_init:  Number of time the k-means algorithm will be run with different 
    centroid seeds. The final results will be the best output of n_init 
    consecutive runs in terms of inertia. (default=10)

    """
    k_means = KMeans(init='k-means++', n_clusters=kopt, n_init=10)
    k_means.fit(fixations_unq)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    
    _,  n_saccades_per_fixation = np.unique(k_means_labels, return_counts=True)
    
    return k_means_cluster_centers, k_means_labels, n_saccades_per_fixation

def sort_cluster_centers_wrt_saliency(smap, k_means_cluster_centers):
    """
    After I get the cluster center estimates, I check their saliency in the 
    original saliency map.
    
    I then sort the estimates from the one with largest saliency, towards the 
    one with smallest saliency.
    
    When I generate synthetic fixation patterns, I start from the begining of 
    this array (my first synthetic fixation most likely to be attended).
    """
    
    saliencies_at_cluster_centers = []
    
    for cc in k_means_cluster_centers:
        saliencies_at_cluster_centers.append(
                smap[int(cc[1])][ int(cc[0])])
        
    
    # argsort sorts in ascending order
    # so I flip the indices array
    inds = np.array( saliencies_at_cluster_centers ).argsort()[::-1] 
    
    k_means_cluster_centers_sorted = []
    for iii in inds:
        k_means_cluster_centers_sorted.append( k_means_cluster_centers[iii] )
        
    return k_means_cluster_centers_sorted
    

def get_pdfs_emp(nsf_wrt_age_motiv_objtype, myrange, mybins):
    """
    This function gets empirical pdf (e.g. of n_saccades_per_fixation).
    
    The shape of output is same as others: pdf_emp_nfixations[a][m][o]
    """
    
    pdf_emp_nsf = init_pdf_emp()
    bin_edges_emp_nsf = init_pdf_emp()

    for ai, a in enumerate(constants.aS):
        for o in constants.OBJECT_TYPES:        
            for mi, m in enumerate( constants.MOTIVATIONS ):
            
                temp = np.hstack(nsf_wrt_age_motiv_objtype[a][m][o])

                if mybins == 0:
                    # if it is not defined use 10
                    mybins = 10
                
                
                if myrange == 0:
                    hist, bin_edges = np.histogram(temp, bins=mybins)
                else:
                    hist, bin_edges = np.histogram(temp, range=myrange, bins=mybins)
                        
                
                bin_size = bin_edges[1] - bin_edges[0]
                pdf = hist / np.sum(hist) / bin_size    
                
                pdf_emp_nsf[a][m][o] = pdf
                bin_edges_emp_nsf[a][m][o] = bin_edges

    return pdf_emp_nsf, bin_edges_emp_nsf

def sample_from_smap(image_orig, smap):
    """
    This function considers smap as a 2D pdf. It randomly samples points from 
    this distribution. These can be considered as synthetic fixations.
    
    The sampling is dense, i.e. I sample too many points. See preferences for 
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
                            size=preferences.N_OVERSAMPLE, replace=False)
    
    temp_random_saccades = np.unravel_index(inds, (H,W))
    
    random_saccades = []
    for (x,y) in zip(temp_random_saccades[0], temp_random_saccades[1]):
        random_saccades.append([y,x])
       
    return random_saccades   
    
def sample_from_emp_distribution(cdf_emp_nfs, bin_edges_emp_nfs, N_RANDOM_SAMPLES):
    """
    Inverse transform sampling (also known as inversion sampling, the inverse 
    probability integral transform, the inverse transformation method, Smirnov 
    transform, universality of the uniform, or the golden rule[1]) is a basic 
    method for pseudo-random number sampling, i.e., for generating sample 
    numbers at random from any probability distribution given its cumulative 
    distribution function. 
    """
    random_samples = init_pdf_emp()
        
        
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:        
            for mi, m in enumerate( constants.MOTIVATIONS ):
                
        
                random_samples[age_range][m][object_type] = sample_from_emp_distribution_small(\
                                                              cdf_emp_nfs[age_range][m][object_type],\
                                                              bin_edges_emp_nfs[age_range][m][object_type],\
                                                              N_RANDOM_SAMPLES)
    return random_samples
        
def sample_from_emp_distribution_small(cdf, bin_edges, N_RANDOM_SAMPLES):
    """
    Inverse transform sampling (also known as inversion sampling, the inverse 
    probability integral transform, the inverse transformation method, Smirnov 
    transform, universality of the uniform, or the golden rule[1]) is a basic 
    method for pseudo-random number sampling, i.e., for generating sample 
    numbers at random from any probability distribution given its cumulative 
    distribution function. 
    """
    random_samples = []
        
    for i in range(N_RANDOM_SAMPLES):
        
        r = random.uniform(0, 1)
        ind = np.argmin( abs(np.subtract( cdf, r) ) ) 
        
        random_samples.append( bin_edges[ind] )
        
    return random_samples


def sample_from_emp_distribution_log_displacement(\
                                 cdf_d_wrt_age_motiv_objtype, bin_edges_emp_d,\
                                 cdf_dx_wrt_age_motiv_objtype, bin_edges_emp_dx,\
                                 cdf_dy_wrt_age_motiv_objtype, bin_edges_emp_dy,\
                                 N_RANDOM_SAMPLES):

    """
    Inverse transform sampling (also known as inversion sampling, the inverse 
    probability integral transform, the inverse transformation method, Smirnov 
    transform, universality of the uniform, or the golden rule[1]) is a basic 
    method for pseudo-random number sampling, i.e., for generating sample 
    numbers at random from any probability distribution given its cumulative 
    distribution function. 
    """
    random_samples_d, random_samples_dx, random_samples_dy = \
    init_pdf_emp(), init_pdf_emp(), init_pdf_emp()
        
        
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:        
            for mi, m in enumerate( constants.MOTIVATIONS ):

        
                random_samples_d[age_range][m][object_type] = sample_from_emp_distribution_small(\
                                                              cdf_d_wrt_age_motiv_objtype[age_range][m][object_type],\
                                                              bin_edges_emp_d[age_range][m][object_type],\
                                                              N_RANDOM_SAMPLES)
                
                        
                random_samples_dx[age_range][m][object_type] = sample_from_emp_distribution_small(\
                                                              cdf_dx_wrt_age_motiv_objtype[age_range][m][object_type],\
                                                              bin_edges_emp_dx[age_range][m][object_type],\
                                                              N_RANDOM_SAMPLES)
                
                random_samples_dy[age_range][m][object_type] = sample_from_emp_distribution_small(\
                                                              cdf_dy_wrt_age_motiv_objtype[age_range][m][object_type],\
                                                              bin_edges_emp_dy[age_range][m][object_type],\
                                                              N_RANDOM_SAMPLES)
                                
    return random_samples_d, random_samples_dx, random_samples_dy


def get_synth_pdf_small(random_samples, \
                   bin_edges):
    """    
    Get a single pdf for a single condition (wrt age, motiv, objtype)
    The input variable (random_samples) belong to either nsaccades or nfixations.
    """
    
    hist, bin_edges  = np.histogram(random_samples, bins = bin_edges)
    
    bin_size = bin_edges[1] - bin_edges[0]
    
    pdf = hist / np.sum(hist) / bin_size    
    
    
    return pdf

def get_synth_pdf(random_samples, bin_edges):
    """
    Get all 3 pdfs (for age, motiv, objtype) regarding a variable.
    The input variable (random_samples) belong to either nsaccades or nfixations.
    """

    pdf_synth = init_pdf_emp()
    
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:
            for mi, m in enumerate( constants.MOTIVATIONS ):
                pdf = get_synth_pdf_small(\
                                  random_samples[age_range][m][object_type], \
                                  bin_edges[age_range][m][object_type] \
                                  )
                   
                pdf_synth[age_range][m][object_type] = pdf
 


    return pdf_synth



def get_synth_pdf_displacement(\
                                        random_samples_dx_log_wrt_age_motiv_objtype,\
                                        random_samples_dy_log_wrt_age_motiv_objtype,\
                                        bin_edges_emp_d_wrt_age_motiv_objtype):
    """
    Plot histogram (for d)
    """
    pdf_synth_d_log_wrt_age_motiv_objtype = init_pdf_emp()
    
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:        
            for mi, m in enumerate( constants.MOTIVATIONS ):

                """
                From dx and dy, produce d
               
                There is no need to preprocess since I never sample nan, inf etc
            
                Compute dist d as  sqrt(dx**2 + dy**2)
                """
                
                random_samples_dx_sq = [(np.exp(x))**2 for x in random_samples_dx_log_wrt_age_motiv_objtype[age_range][m][object_type]]
                random_samples_dy_sq = [(np.exp(x))**2 for x in random_samples_dy_log_wrt_age_motiv_objtype[age_range][m][object_type]]
                random_samples_d_log= np.log( np.sqrt(np.add( random_samples_dx_sq , \
                                                        random_samples_dy_sq ))   )
            
            
            
                pdf_synth_d_log_wrt_age_motiv_objtype[age_range][m][object_type]  = \
                get_synth_pdf_small(random_samples_d_log,\
                                    bin_edges_emp_d_wrt_age_motiv_objtype[age_range][m][object_type])
                
    return pdf_synth_d_log_wrt_age_motiv_objtype





def get_cdf_emp(pdf_emp):
    """
    Compute cdf
    """    
    cdf_emp = init_pdf_emp()
    
    for ai, age_range in enumerate(constants.AGE_RANGES):
        for object_type in constants.OBJECT_TYPES:
            for mi, m in enumerate( constants.MOTIVATIONS ):
                cdf = np.cumsum(pdf_emp[age_range][m][object_type])
                cdf = cdf / np.max(cdf)
                
                cdf_emp[age_range][m][object_type] = cdf

    return cdf_emp




def init_r2center_2D():
    """
    This function initializes the 2D r2center array.
    1st dimension is participant and 2nd dimension is time.
    
    Not all participants have the same number of samples. To account for that, 
    I allow the number of samples to be between 90 and 100. See below for 
    reasons of this choice. 
    """
    r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype = \
    {}, {}, {}
    
    for age_range in constants.AGE_RANGES:
        r2center_2d_wrt_age[age_range] = []
            
    for m in constants.MOTIVATIONS:
          r2center_2d_wrt_motiv[m] = []
        
    for objtype in preferences.OBJECT_TYPES_INTEREST:
        r2center_2d_wrt_objtype[objtype] = []
        

    
    return r2center_2d_wrt_age, r2center_2d_wrt_motiv, r2center_2d_wrt_objtype



def pad_fixation_array(r2center):
    """
    This function pads the fixation array to NMAX_FIXATIONS. 
    
    Since the fixation target is shown for 1080 msec (see constants),
    we expect to have roughly 97 samples over each fixation target.
    
    Nevertheless, the sensor does not always sample 90 samples in a sec. If the 
    participant blinks, looks away etc, we collect less than 90 samples per sec.

    If the number of samples between 90 and 100, we align the arrays by simply 
    padding with 0s. 
    
    There is actually some discrepancy here. Because the missing samples can 
    be missed at any point in time. But we assume that they are all missed at the 
    end. 
    
    It  does not matter so much because I take array with length 90 shortest. So
    even though things shift a little in time, they definitely do not shift more 
    than 0.07 sec. Actually, in most cases it is not more than 0.02-0.03 msec 
    """
    for i in range(preferences.NMAX_FIXATIONS - len(r2center)):
        r2center.append(0)
        
    return r2center

def get_r2center_medians_wrt_age(r2center_2d):
    """
    This function gets the median distance of the gaze samples to the center of 
    the fixation target for each age range, separately.
    """
    r2center_medians_wrt_age = {}
    
    for age_range in constants.AGE_RANGES:
        r2center_medians_wrt_age[age_range] = []
        
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """
            temp = [r2center[n] for r2center in r2center_2d[age_range]]
            r2center_medians_wrt_age[age_range].append(np.median(temp))
            

            
    return r2center_medians_wrt_age

def get_r2center_medians_wrt_motiv(r2center_2d_wrt_motiv):
    """
    This function gets the median distance of the gaze samples to the center of 
    the fixation target for each motivation, separately.
    """
    
    r2center_medians_wrt_motiv = {}

    for m in constants.MOTIVATIONS:
        
        r2center_medians_wrt_motiv[m] = []
 
        r2center_2d = r2center_2d_wrt_motiv[m]
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """
            temp = [r2center[n] for r2center in r2center_2d]
            r2center_medians_wrt_motiv[m].append(np.median(temp))

        
    return r2center_medians_wrt_motiv

def get_r2center_medians_wrt_objtype(r2center_2d_wrt_objtype):
    """
    This function gets the median distance of the gaze samples to the center of 
    the fixation target for each object type, separately.
    """
    r2center_medians_wrt_objtype = {}
    
    for o in preferences.OBJECT_TYPES_INTEREST:
        r2center_medians_wrt_objtype[o] = []
        r2center_2d = r2center_2d_wrt_objtype[o]
        for n in range(preferences.NMAX_FIXATIONS): 
            """ 
            Median accounts for the entire data
            """            
            temp = [r2center[n] for r2center in r2center_2d]
            r2center_medians_wrt_objtype[o].append(np.median(temp))
     
    
    return r2center_medians_wrt_objtype


