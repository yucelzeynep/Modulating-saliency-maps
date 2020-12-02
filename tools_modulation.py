#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:03:43 2020

@author: zeynep
"""
import numpy as np
import random
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import tools_saliency as stools

import constants
from importlib import reload
reload(constants)

import preferences
reload(preferences)
    
def init_fixations_eco_random_inv():
    """
    I generate 3 kinds of fixation patterns: eclogical, random and from the 
    complementary map.
    
    Ecological refers to the fact that the fixations resemble those of a human, 
    i.e. subject to saccadic movements as well.
    
    Random is simply a arbitrary pixel drawn from a 2D pdf, whcih is approximated 
    based on the saliency map. 
    
    Inverse is the same as the above, but the approximating function is the 
    complement of that.
    """
    fixations_eco_random_inv = {}
    
    for a in (constants.AGE_RANGES):
        fixations_eco_random_inv[a] = {}
        for m in constants.MOTIVATIONS:      
            fixations_eco_random_inv[a][m] = {}
            for o in ( constants.OBJECT_TYPES ):
                fixations_eco_random_inv[a][m][o] = {'image_fnames':[],\
                                             'fixations_eco':[],\
                                             'fixations_random':[],\
                                             'fixations_inv':[]}

    return fixations_eco_random_inv
    
    
def init_fmaps_eco_random_inv():
    """
    This function loads the fixation patterns saved in the data structure (initialized
    as above in init_fixations_eco_random_inv ) and build a variable with the 
    same structure involving the fixation maps. 
    """
    fmaps_eco_random_inv = {}
    
    for age_range in constants.AGE_RANGES:  
        fmaps_eco_random_inv[age_range] = {}      
        for m in constants.MOTIVATIONS:            
            fmaps_eco_random_inv[age_range][m] = {}     
            for object_type in np.sort( constants.OBJECT_TYPES):
                fmaps_eco_random_inv[age_range][m][object_type] =  {'image_fnames':[],\
                                             'fmaps_eco':[],\
                                             'fmaps_random':[],\
                                             'fmaps_inv':[]}

    return fmaps_eco_random_inv
    
    
    
def sample_from_emp_distribution(cdf, bin_edges):
    """
    Inverse transform sampling (also known as inversion sampling, the inverse 
    probability integral transform, the inverse transformation method, Smirnov 
    transform, universality of the uniform, or the golden rule[1]) is a basic 
    method for pseudo-random number sampling, i.e., for generating sample 
    numbers at random from any probability distribution given its cumulative 
    distribution function. 
    """
    # here nsf stands for number of saccades or fixations (ns or nf)
    random_nsf = 0
    
    while random_nsf == 0:
        
        r = random.uniform(0, 1)
        ind = np.argmin( abs(np.subtract( cdf, r) ) ) 
        
        random_nsf = int(  bin_edges[ind]  )# just in case cast to int
        
    return random_nsf



def sample_from_emp_distribution_log_displacement(cdf, bin_edges):
    """
    Again inverse transform sampling, this time applied on the log of the 
    displacement.
    """
    r = random.uniform(0, 1)
    ind = np.argmin( abs(np.subtract( cdf, r) ) ) 
    
    displacement_between_fixations_eco_log = bin_edges[ind] 
    
    displacement_between_fixations_eco = np.exp( displacement_between_fixations_eco_log )
        
    return displacement_between_fixations_eco
    

def get_nsaccades_nfixations(a,m,o,\
                             cdf_emp_nsaccades, \
                             bin_edges_emp_nsaccades,\
                             cdf_emp_nfixations, \
                             bin_edges_emp_nfixations):
    """  
    This functions generates a sequnece os fixations which mimic the ecological 
    fixatiosn (of a human).
    
    I sample N_SAMPLING_MAX times nsaccades_eco, and then I sample as many 
    nfixations_eco as nsaccades_eco.
    
    Finally, I compute ntot_fixations for each case and choose the pair
    (nsaccades_eco, nfixations_eco), which gives the closest total number
    of gaze samples to 250.
    
    Basically, I need sth around 250 due to the display duration of the image 
    (which is 2.5 sec).
    
    This approach is quite different than the synthetic distribution in 
    get_synth_distributions. Because there we do not need a direct relation 
    between nsaccades_eco and nfixations_eco.
    """
    
    temp_nsaccades_eco = []    
    temp_nfixations_eco = []
    temp_ntot_fixations = []
    
    for ttt in range(preferences.N_SAMPLING_MAX):
        
        ntot_fixations = 0
        
        nsaccades_eco = 0 # this cannot be zero, so keep on sampling until you get nonzero
        while nsaccades_eco == 0:
            nsaccades_eco = sample_from_emp_distribution(\
                                  cdf_emp_nsaccades[a][m][o],\
                                  bin_edges_emp_nsaccades[a][m][o])
        
        dummy = []
        for r in range( int(nsaccades_eco) ):
    
            nfixations_eco = 0 # this cannot be zero, so keep on sampling until you get nonzero
            while nfixations_eco == 0:
                # get number of fixations for this time
                nfixations_eco = sample_from_emp_distribution(\
                                    cdf_emp_nfixations[a][m][o],\
                                    bin_edges_emp_nfixations[a][m][o])  
            
            dummy.append(nfixations_eco)
            ntot_fixations +=  nfixations_eco 
            
        temp_nfixations_eco.append(nsaccades_eco)
        temp_nsaccades_eco.append(dummy)
        temp_ntot_fixations.append(ntot_fixations)
        
    """
    Which cobination (ns+nf) give the closest number to 250?
    Careful with the absolute value
    """
    temp_diff_with_250 = [x - 250 for x in temp_ntot_fixations]
    ind = np.argmin( np.abs(temp_diff_with_250))
    
    nsaccades_eco = temp_nfixations_eco[ind]
    nfixations_eco = temp_nsaccades_eco[ind]
    ntot_fixations = temp_ntot_fixations[ind]
    
    return nsaccades_eco, nfixations_eco, ntot_fixations
    
    
        
def get_indices_on_a_circle(current_fixation,\
                            displacement_between_fixations_estimation):
    """
    This function returns a set indices of potential fixation points around an 
    initial fixation point.
    
    From current_fixation, we want to move away by the amount of 
    displacement_between_fixations_estimation. So I get the indices of all 
    pixels satisfying this condition.
    
    I will use another function to choose the next fixation. that will be the 
    location with highest saliency over this circle. 
    """
    x = np.arange(0, constants.IMAGE_WIDTH)
    y = np.arange(0, constants.IMAGE_HEIGHT)
    temp_image = np.zeros((y.size, x.size))
    
    # I merged mask and array
    # see example code for further clarity
    temp_image[(x[np.newaxis,:]- current_fixation[0] )**2 +\
    (y[:,np.newaxis]-current_fixation[1])**2 < displacement_between_fixations_estimation**2] = 123.
                
    # take gradient and solve for indices
    laplacian = cv2.Laplacian(temp_image,cv2.CV_64F)
    (next_fixation_indx_candidates, next_fixation_indy_candidates) = np.where(laplacian)
        
    return next_fixation_indx_candidates, next_fixation_indy_candidates



def get_next_fixation(current_fixation, smap_with_supressions, \
                     next_fixation_indx_candidates, next_fixation_indy_candidates):
    """
    I get the saliencies of the pixels over the circle and store them in an 
    array.
    
    Previously I used to retrive the index on the circle, which corresponds to 
    the maximum saliency.
    
    But I changed my approach, since this resulted sometimes in  unnatural 
    eye movements, such as moving over horizontal line or vertical line.  
    
    I decided to use Fitness proportionate selection. See below for explanation
    """
    saliencies_at_next_fixation_candidates = []
    
  
    for indx,indy in zip(next_fixation_indx_candidates, next_fixation_indy_candidates):
        saliencies_at_next_fixation_candidates.append( smap_with_supressions[indx][indy] )
            
    """
    Here I do Fitness proportionate selection (also known as wheel of fortune,
    roulette wheel selection etc)
    https://en.wikipedia.org/wiki/Fitness_proportionate_selection  
    
    It chooses a pixel over the circle assuming that the saliencies are the 
    probabilities. It is similar to Smirnov transform but that one is 
    absolutely necessary in analytical functions, whereas this on simply 
    applies to -discrete- vectors.
    """
    temp = []
    while len(temp) == 0:
        sal_cumsum = np.cumsum(saliencies_at_next_fixation_candidates)
        sal_cumsum_norm = (sal_cumsum - np.min(sal_cumsum)) / (np.max(sal_cumsum) - np.min(sal_cumsum))
        myr = random.uniform(0, 1)
        temp = [i for i, val in enumerate(sal_cumsum_norm) if val >= myr]
    min_ind = np.min( temp )

    next_fixation = [ int(next_fixation_indy_candidates[min_ind]), \
                     int(next_fixation_indx_candidates[min_ind])]
    
    """
    This part emulates inhibition of return. Namely, when you visit one pixel
    on the image, you are less likely to visit it again, and for avoiding 
    repetitive gazing, I reduce the saliency of that pixel so that it does not
    appear once more among the subsequent ecological fixations.
    """
    smap_with_supressions[next_fixation_indx_candidates[min_ind] ][ next_fixation_indy_candidates[min_ind] ] = 0
           
             
    return next_fixation, smap_with_supressions

                  
def get_fixations_eco(myobject, cluster_center_candidates, image_fname,
                      a,m,o,\
                      cdf_emp_nsaccades, \
                      bin_edges_emp_nsaccades,\
                      cdf_emp_nfixations, \
                      bin_edges_emp_nfixations,\
                      cdf_d_wrt_age_motiv_objtype, \
                      cdf_dx_wrt_age_motiv_objtype, \
                      cdf_dy_wrt_age_motiv_objtype,\
                      bin_edges_emp_d_wrt_age_motiv_objtype,\
                      bin_edges_emp_dx_wrt_age_motiv_objtype,\
                      bin_edges_emp_dy_wrt_age_motiv_objtype):
    """
    First I sample from empirical distributions to get (i) nsaccades_eco and 
    (ii) nfixations_eco.
        
    Then, I start from the cluster center with highest saliency in the list as 
    the initial fixation. 
    
    Later, I sample from empirical distribution of logarithm of displacement.
     and take inverse logarithm. 
    
    I start moving from the cluster center to another pixel that lies in a 
    distance of displacement_between_fixations_eco. After visint that pixel, I 
    apply inhibition of return on the saliency map, so that that pixel will not
    be visited again.
    
    I keep on sampling a displacement_between_fixations_eco and moving to a 
    successive pixel as many times as nfixations_eco.
    
    Then I restart from the next cluster center in the list and repeat this 
    nsaccades_eco times. 
    """   
    
    fixations_eco = []
        
    nsaccades_eco, nfixations_eco, ntot_fixations = get_nsaccades_nfixations(\
                                                                            a,m,o,\
                                                                            cdf_emp_nsaccades, \
                                                                            bin_edges_emp_nsaccades,\
                                                                            cdf_emp_nfixations, \
                                                                            bin_edges_emp_nfixations)
 
    
    smap_with_supressions = myobject.smap[preferences.BASELINE_SALIENCY_TYPE].copy() 
            
    for r in range( int(nsaccades_eco) ):
        
        # the first fixation is the cluster center candidate
        cluster_center = cluster_center_candidates[o][image_fname][r]
        fixations_eco.append( list(cluster_center) )
        
        current_fixation = list( fixations_eco[-1] )

        for s in range( nfixations_eco[r] ):
            
            displacement_between_fixations_eco = \
            sample_from_emp_distribution_log_displacement(\
            cdf_d_wrt_age_motiv_objtype[a][m][o],\
            bin_edges_emp_d_wrt_age_motiv_objtype[a][m][o])
            
            next_fixation_indx_candidates = [] 
            while len(next_fixation_indx_candidates) == 0:
                next_fixation_indx_candidates, next_fixation_indy_candidates = \
                get_indices_on_a_circle(current_fixation, \
                                        displacement_between_fixations_eco)
            
            next_fixation, smap_with_supressions = \
            get_next_fixation(current_fixation, smap_with_supressions,\
                             next_fixation_indx_candidates, next_fixation_indy_candidates)
            
            
            current_fixation = next_fixation
            
            fixations_eco.append(next_fixation)

            
    return fixations_eco
   
def get_fixations_ri(image_orig, smap):
    """
    I choose N_FIXATIONS_RANDOM for each image, from the set of all pixels, 
    without replacement (no pixel appears more than once).
    
    The function can be used to get fixations from smap randomly (i.e. not in 
    the ecological way) or from the complement of smap (smap_inv)
    """
    
    if len(image_orig.shape) == 2:
        H, W = image_orig.shape
    else:
        H, W, _ = image_orig.shape
            
    smap_normalized = smap / smap.sum() 
    
    """"
    Make n random selections from the flattened pmf without replacement
    Whether you want replacement, depends on your application (I dont want)
    """
    inds = np.random.choice(\
                            np.arange(W*H),\
                            p=smap_normalized.reshape(-1), \
                            size=preferences.N_FIXATIONS_RANDOM, replace=False)
    
    temp_fixations = np.unravel_index(inds, (H,W))
    
    fixations_ri = [] # ri stand for random or inv
    for (x,y) in zip(temp_fixations[0], temp_fixations[1]):
        fixations_ri.append([y,x])

    return fixations_ri


    

    

def init_rate_of_gaze():
    """       
    I have three keys as:
        func
        manip
        neither
    
    Functional part is the part where humans grasp and operate an object.
    Manipulative part is the end-effector, where the object realizes its purpose.
    Neither is outside any of these two polygons.
    """
    rate_of_gaze_wrt_age, \
    rate_of_gaze_wrt_motiv, \
    rate_of_gaze_wrt_objtype = {}, {}, {}
    
    for age_range in constants.AGE_RANGES:        
        rate_of_gaze_wrt_age[age_range] = {'func': [], 'manip':[], 'neither':[]}
    
    for m in constants.MOTIVATIONS:            
        rate_of_gaze_wrt_motiv[m] = {'func': [], 'manip':[], 'neither':[]}
        
    for object_type in np.sort( preferences.OBJECT_TYPES_INTEREST):
        rate_of_gaze_wrt_objtype[object_type] = {'func': [], 'manip':[], 'neither':[]}
        
    return rate_of_gaze_wrt_age, \
    rate_of_gaze_wrt_motiv, \
    rate_of_gaze_wrt_objtype
   

def get_rate_of_gaze(myobject,\
                               image_fixations,\
                               rate_of_gaze_wrt_age, \
                               rate_of_gaze_wrt_motiv, \
                               rate_of_gaze_wrt_objtype, \
                               age_range,\
                               motiv,\
                               object_type):
    """
    For each fixation, decide which polygon it falls in and count the number of
    samples for each viewing. 
    """
    samples_in_func, samples_in_manip, samples_in_neither = [], [], []

    # build polygon object          
    temp_poly = myobject.polygon_functional
    polygon_func = Polygon([tuple(temp_poly[0]), tuple(temp_poly[1]),\
                   tuple(temp_poly[2]), tuple(temp_poly[3])])
    
    temp_poly = myobject.polygon_manipulative
    polygon_manip = Polygon([tuple(temp_poly[0]), tuple(temp_poly[1]),\
                   tuple(temp_poly[2]), tuple(temp_poly[3])])
    
    # count each point
    for p in image_fixations:
        point = Point(p[0], p[1])
        if polygon_func.contains(point):
            samples_in_func.append( [p[0], p[1]])
        elif polygon_manip.contains(point):
            samples_in_manip.append( [p[0], p[1]] )
        else:
            samples_in_neither.append( [p[0], p[1]] )
            

    
    rate_of_gaze_wrt_age[age_range]['func'].append ( len(samples_in_func) )
    rate_of_gaze_wrt_age[age_range]['manip'].append ( len(samples_in_manip) )
    rate_of_gaze_wrt_age[age_range]['neither'].append ( len(samples_in_neither) )    
    
    rate_of_gaze_wrt_motiv[motiv]['func'].append ( len(samples_in_func) )
    rate_of_gaze_wrt_motiv[motiv]['manip'].append ( len(samples_in_manip) )
    rate_of_gaze_wrt_motiv[motiv]['neither'].append ( len(samples_in_neither) ) 
    
    rate_of_gaze_wrt_objtype[object_type]['func'].append ( len(samples_in_func) )
    rate_of_gaze_wrt_objtype[object_type]['manip'].append ( len(samples_in_manip) )
    rate_of_gaze_wrt_objtype[object_type]['neither'].append ( len(samples_in_neither) )     
    
    return samples_in_func, \
    samples_in_manip, \
    samples_in_neither, \
    rate_of_gaze_wrt_age, \
    rate_of_gaze_wrt_motiv, \
    rate_of_gaze_wrt_objtype
   
  
    
                      
def get_rate_of_gaze_stats(rate_of_gaze_wrt_sth):
    """
    Get the mean number of fixations in different parts of the image (corresponding
    to different parts of the object)
    """
       
    rate_of_gaze_means = {}
    
    for k in rate_of_gaze_wrt_sth.keys():
        rate_of_gaze_means[k] = {}
        
        for pi, part in enumerate(constants.OBJECT_PARTS):
            
            mean_temp = np.mean( rate_of_gaze_wrt_sth[k][part] ) 
            
            rate_of_gaze_means[k][part] = mean_temp
       
    
    return rate_of_gaze_means
    
    
    

def init_r_fix_mats():
    """
    Initialialize saliency distance matrix contrasting motivations and 
    image_fname.
    
    The shape is as follows:
        r_fix_wrt_amop[age_range][m][object_type] 
        
    And then I have three keys as:
        fund
        manip
        neither
    
    Func is grip/handle etc\
    useful part of the object.
    Neither is outside any of these two polynomials.
    
    """      
        
    r_fix_wrt_amop = {}
    
    for a in constants.AGE_RANGES:
        
        r_fix_wrt_amop[a] = {}
        
        for m in constants.MOTIVATIONS:
            
            r_fix_wrt_amop[a][m] = {}
            
            for object_type in np.sort( preferences.OBJECT_TYPES_INTEREST):
                
                r_fix_wrt_amop[a][m][object_type] = {'func': [], 'manip':[], 'neither':[]}

    
    return  r_fix_wrt_amop
   
def remove_blank_target_fixations(fixations_emp, all_image_fnames):
    """
    The empirical fixatipns invove also those over the blank image and fixation
    target image. Here I remove the fixatios over these images and leave only the fixations over 
    th object images.
    """
    temp = []
    for i, image_fname in enumerate(all_image_fnames):
            
        if image_fname != constants.TARGET_IMAGE_FNAME and\
                            image_fname != constants.BLANK_IMAGE_FNAME:
                                temp.append( fixations_emp[i] )

    return temp


def allocate_fixations_wrt_amop(myobject, \
                                   image_fixations, \
                                   r_fix_wrt_amop,
                                   a, m, o):
    """
    For each fixation, dcide which polygon is falls in and count the points
    for each image. 
    """
    fix_in_func, fix_in_manip, fix_in_neither = [], [], []
    
    

    # build polygon object          
    temp_poly = myobject.polygon_functional
    polygon_func = Polygon([tuple(temp_poly[0]), tuple(temp_poly[1]),\
                   tuple(temp_poly[2]), tuple(temp_poly[3])])
    
    temp_poly = myobject.polygon_manipulative
    polygon_manip = Polygon([tuple(temp_poly[0]), tuple(temp_poly[1]),\
                   tuple(temp_poly[2]), tuple(temp_poly[3])])
    
    # count each point
    for p in image_fixations:
        point = Point(p[0], p[1])
        if polygon_func.contains(point):
            fix_in_func.append( [p[0], p[1]])
        elif polygon_manip.contains(point):
            fix_in_manip.append( [p[0], p[1]] )
        else:
            fix_in_neither.append( [p[0], p[1]] )
            
    r_fix_wrt_amop[a][m][o]['func'].append ( len(fix_in_func) / len(image_fixations))
    r_fix_wrt_amop[a][m][o]['manip'].append ( len(fix_in_manip)/ len(image_fixations) )
    r_fix_wrt_amop[a][m][o]['neither'].append ( len(fix_in_neither)/ len(image_fixations) )    
            
    
    return  r_fix_wrt_amop 
   
  
    
                      
def get_r_fixation_stats(r_fix_wrt_amop):
        
    r_fix_stats_wrt_ap = {}
    for a in constants.AGE_RANGES:
        r_fix_stats_wrt_ap[a] = {}
        
    r_fix_stats_wrt_mp = {}
    for m in constants.MOTIVATIONS:
        r_fix_stats_wrt_mp[m] = {}
        
    r_fix_stats_wrt_op = {}
    for o in constants.OBJECT_TYPES:
        r_fix_stats_wrt_op[o] = {}
                          

    ###########################################################
    temp_ap = {}
    for a in constants.AGE_RANGES:
        temp_ap[a] = {}
        for p in constants.OBJECT_PARTS:
            temp_ap[a][p] = []
            
    for a in constants.AGE_RANGES:
        for m in constants.MOTIVATIONS:
            for o in np.sort( preferences.OBJECT_TYPES_INTEREST):
                for p in constants.OBJECT_PARTS:
                    temp_ap[a][p].extend( r_fix_wrt_amop[a][m][o][p] )
                    
    for a in constants.AGE_RANGES:
        for p in constants.OBJECT_PARTS:
            r_fix_stats_wrt_ap[a][p] = np.mean(temp_ap[a][p])                   
                
    ###########################################################
    temp_mp = {}
    for m in constants.MOTIVATIONS:
        temp_mp[m] = {}
        for p in constants.OBJECT_PARTS:
            temp_mp[m][p] = []
            
    for a in constants.AGE_RANGES:
        for m in constants.MOTIVATIONS:
            for o in np.sort( preferences.OBJECT_TYPES_INTEREST):
                for p in constants.OBJECT_PARTS:
                    temp_mp[m][p].extend( r_fix_wrt_amop[a][m][o][p] )
                    
    for m in constants.MOTIVATIONS:
        for p in constants.OBJECT_PARTS:
            r_fix_stats_wrt_mp[m][p] = np.mean(temp_mp[m][p])                   
                    
    ###########################################################
    temp_op = {}
    for o in constants.OBJECT_TYPES:
        temp_op[o] = {}
        for p in constants.OBJECT_PARTS:
            temp_op[o][p] = []
            
    for a in constants.AGE_RANGES:
        for m in constants.MOTIVATIONS:
            for o in np.sort( preferences.OBJECT_TYPES_INTEREST):
                for p in constants.OBJECT_PARTS:
                    temp_op[o][p].extend( r_fix_wrt_amop[a][m][o][p] )
                    
    for o in constants.OBJECT_TYPES:
        for p in constants.OBJECT_PARTS:
            r_fix_stats_wrt_op[o][p] = np.mean(temp_op[o][p])    
    
    
    
    return r_fix_stats_wrt_ap, r_fix_stats_wrt_mp, r_fix_stats_wrt_op    




def init_supp_maps_eco_random_inv():
    """
    Init smaps supp and modif
    
    One for each participant
    """            
    smaps_supp_modif = {}
    
    for a in (constants.AGE_RANGES):
        smaps_supp_modif[a] = {}
        for m in constants.MOTIVATIONS:      
            smaps_supp_modif[a][m] = {}
            for o in ( constants.OBJECT_TYPES ):                           
                smaps_supp_modif[a][m][o] = {\
                                        'image_fnames': [],\
                                        'supp_map_eco_pos': [],\
                                        'supp_map_random_pos': [],\
                                        'supp_map_neg': []}
    
    return smaps_supp_modif




def mask_fixations_with_poly(fixations, poly):
    """
    Mask a given set of fixations (eco or inv) with a given polygon 
    The polygon is :
        poly_pos (i.e. manip) for eco
        poly_neg (i.e. func) for inv
    """
    
    fixations_masked = []
    
    polygon = Polygon([tuple(poly[0]), tuple(poly[1]),\
                       tuple(poly[2]), tuple(poly[3])])
    

    for f in fixations:
        point = Point(f[0], f[1])
        if polygon.contains(point):
            fixations_masked.append(f)
            
    return fixations_masked


def get_supp_maps(fixations_eco_masked, fixations_random_masked, fixations_inv_masked):
    """
    Get fmaps from fixations and take their difference
    """
    H, W = constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH
    
    supp_map_eco_pos = stools.Fixpos2Densemap(fixations_eco_masked, W, H)
    supp_map_random_pos = stools.Fixpos2Densemap(fixations_random_masked, W, H)
    supp_map_neg = stools.Fixpos2Densemap(fixations_inv_masked, W, H)

    
    return supp_map_eco_pos, supp_map_random_pos, supp_map_neg


def superimpose_maps_f(smap_orig, \
                       supp_map_pos, \
                       supp_map_neg,\
                       modif_coefs, \
                       a, m, o):
    """
    Scale up manipulative part
    scale down functional part
    
    Actually up or down is decided by sign of modif_coef 
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
#    mask_pos = (supp_map_pos>0) * 255    
#    mask_pos = mask_pos.astype(np.uint8)    
#    
#     # this is usually positive so i call it amplify
#    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
#    
#    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
#    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
#    smap_modif[smap_modif> 255] = 255
                        
    mask_neg = (supp_map_neg>0) * 255    
    mask_neg = mask_neg.astype(np.uint8)    
    
    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
    
    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
    smap_modif[smap_modif> 255] = 255
   
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif

def superimpose_maps_m(smap_orig, \
                       supp_map_pos, \
                       supp_map_neg, \
                       modif_coefs, \
                       a, m, o):
    """
    Scale up manipulative part
    scale down functional part
    
    Actually up or down is decided by sign of modif_coef 
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
    mask_pos = (supp_map_pos>0) * 255    
    mask_pos = mask_pos.astype(np.uint8)    
    
     # this is usually positive so i call it amplify
    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
    
    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
    smap_modif[smap_modif> 255] = 255
                        
#    mask_neg = (supp_map_neg>0) * 255    
#    mask_neg = mask_neg.astype(np.uint8)    
#    
#    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
#    
#    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
#    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
#    smap_modif[smap_modif> 255] = 255
   
  
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif
    

def superimpose_maps_fm(smap_orig, \
                        supp_map_pos, \
                        supp_map_neg, \
                        modif_coefs, \
                        a,m,o):
    """
    Scale up manipulative part
    scale down functional part
    """           
    smap_modif = smap_orig.copy().astype(np.float)
    
    mask_pos = (supp_map_pos>0) * 255    
    mask_pos = mask_pos.astype(np.uint8)    
    
     # this is usually positive so i call it amplify
    coef_amplify = 1 + np.sign(modif_coefs[a][m][o]['manip']) * modif_coefs[a][m][o]['manip']**2
    
    smap_modif[mask_pos>0] = (1-coef_amplify)* smap_orig.astype(np.float)[mask_pos>0] + \
    coef_amplify * supp_map_pos.astype(np.float)[mask_pos>0] 
    smap_modif[smap_modif> 255] = 255
                        
    mask_neg = (supp_map_neg>0) * 255    
    mask_neg = mask_neg.astype(np.uint8)    
    
    coef_attenuate = 1 + np.sign(modif_coefs[a][m][o]['func']) * modif_coefs[a][m][o]['func']**2
    
    smap_modif[mask_neg>0] =  (1-coef_attenuate)* smap_orig.astype(np.float)[mask_neg>0] + \
    coef_attenuate * supp_map_neg.astype(np.float)[mask_neg>0] 
    smap_modif[smap_modif> 255] = 255
   
    smap_modif = smap_modif.astype(np.uint8)
            
    return smap_modif



    