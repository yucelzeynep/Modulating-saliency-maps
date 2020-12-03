#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:33:21 2020

@author: zeynep
"""

import numpy as np
import math

from importlib import reload


import constants
reload(constants)

import preferences
reload(preferences)
   
 

def init_metrics():
    
    metrics_emp2orig = {}
    metrics_emp2modif_eco = {}
    metrics_emp2modif_random = {}
    
    for a in constants.AGE_RANGES:
        metrics_emp2orig[a], metrics_emp2modif_eco[a], metrics_emp2modif_random[a] = \
            {}, {}, {}
        for m in constants.MOTIVATIONS:
            metrics_emp2orig[a][m],metrics_emp2modif_eco[a][m], \
                metrics_emp2modif_random[a][m] = {}, {}, {}
            for o in constants.OBJECT_TYPES:
                metrics_emp2orig[a][m][o],metrics_emp2modif_eco[a][m][o],\
                    metrics_emp2modif_random[a][m][o] = {}, {}, {}
                for sm in preferences.SALIENCY_METRICS:
                    metrics_emp2orig[a][m][o][sm] = []
                    metrics_emp2modif_eco[a][m][o][sm] = {'f':[], 'm':[],'fm':[]}
                    metrics_emp2modif_random[a][m][o][sm] = {'f':[], 'm':[],'fm':[]}

    return metrics_emp2orig, metrics_emp2modif_eco, metrics_emp2modif_random
                    

def compute_metric(saliency_metric, smap, fmap, baseline_center_prior):
 
    if saliency_metric is 'AUC_JUDD':
        return auc_judd(smap, fmap)
    elif saliency_metric is  'AUC_BORJI':
        return auc_borji(smap, fmap)
    elif saliency_metric is  'AUC_SHUFF':
        return auc_shuff(smap, fmap)
    elif saliency_metric is  'NSS':
        return nss(smap, fmap)
    elif saliency_metric is  'INFOGAIN':
        return infogain(smap, fmap, baseline_center_prior)
    elif saliency_metric is  'SIM':
        return similarity(smap, fmap)
    elif saliency_metric is  'CC':
        return cc(smap, fmap)
    elif saliency_metric is  'KLDIV':
        return kldiv(smap, fmap)
    else:
        print('No such metric as ', saliency_metric)
        return 0
        
    
    
def GaussianMask( sizex,sizey, sigma=33, center=None,fix=1):
    """
    For building a patch, which follows a Gaussian distribution
    
    sizex  : mask width
    sizey  : mask height
    sigma  : Sdt of the Gaussian
    center : mean of the Gaussian
    fix    : max of the Gaussian
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)



def Fixpos2Densemap(fix_arr, width, height):
    """
    Represents the gaze distribution as a heat map
    
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    """
    
    heatmap = np.zeros((height,width), np.float32)
    for n_subject in range(len(fix_arr)):
        # since I go 1-by-1 I have a fixation number of 1 at every step
        heatmap += GaussianMask(width, height, 33, (fix_arr[n_subject][0],fix_arr[n_subject][1]), 1)

    # Normalization
    if np.amax(heatmap) > 0:
        heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    # for color
    # if you want gray level omit the below
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def build_fmap_from_fixations(fixations, window_title_fmap):
    """
    Build fixation maps. This is necessary to compute the saliency difference 
    (or similarity) matrix.
    """

    fmap = Fixpos2Densemap(fixations, constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
                                      
    return fmap

"""
For the below, please refer to 
https://github.com/tarunsharma1/saliency_metrics
who translates from 
https://github.com/cvzoya/saliency/tree/master/code_forMetrics 

For installing tqdm, 
sudo python3 -m pip install tqdm
"""

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255

def auc_judd(s_map,gt):
    """
    Area under the curve by Judd et al.
    """
	# ground truth is discrete, s_map is continous and normalized
	#gt = discretize_gt(gt)
	# thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0,gt.shape[0]):
        for k in range(0,gt.shape[1]):
            if gt[i][k]>0:
                thresholds.append(s_map[i][k])

	
    num_fixations = np.sum(gt)
	# num fixations is no. of salience map values at gt >0


    thresholds = sorted(set(thresholds))
	
	#fp_list = []
	#tp_list = []
    area = []
    area.append((0.0,0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map>=thresh] = 1.0
        assert np.max(gt)==1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(s_map)==1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
        tp = num_overlap/(num_fixations*1.0)
		
		# total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
		# this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
		
        area.append((round(tp,4),round(fp,4)))
		#tp_list.append(tp)
		#fp_list.append(fp)

	#tp_list.reverse()
	#fp_list.reverse()
    area.append((1.0,1.0))
	#tp_list.append(1.0)
	#fp_list.append(1.0)
	#print tp_list
    area.sort(key = lambda x:x[0])
    tp_list =  [x[0] for x in area]
    fp_list =  [x[1] for x in area]
    return np.trapz(np.array(tp_list),np.array(fp_list))



def auc_borji(s_map,gt,splits=100,stepsize=0.1):
    """
    Area under the curve by Borji et al.
    """
	#gt = discretize_gt(gt)
    num_fixations = int(np.sum(gt))

    num_pixels = s_map.shape[0]*s_map.shape[1]
    random_numbers = []
    for i in range(0,splits):
        temp_list = []
        for k in range(0,num_fixations):
            temp_list.append(np.random.randint(num_pixels))
            random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k%s_map.shape[0]-1][ int(k/s_map.shape[0]) ]) # by z
		# in these values, we need to find thresholds and calculate auc
        thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        r_sal_map = np.array(r_sal_map)

		# once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0,0.0))
        for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map>=thresh] = 1 # from 1.0 to 1 by z
            num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0] # transpose by z
            tp = num_overlap/(num_fixations*1.0)
			
			#fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

            area.append((round(tp,4),round(fp,4)))
		
        area.append((1.0,1.0))
        area.sort(key = lambda x:x[0])
        tp_list =  [x[0] for x in area]
        fp_list =  [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))
	
    return np.mean(aucs)


def auc_shuff(s_map,gt,other_map,splits=100,stepsize=0.1):
    """
    Area under the curve
    """
	#gt = discretize_gt(gt) # comment out by z
	#other_map = discretize_gt(other_map) # comment out by z

    num_fixations = np.sum(gt)
	
    x,y = np.where(other_map==1)
    other_map_fixs = []
    for j in zip(x,y):
        other_map_fixs.append(j[0]*other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind==np.sum(other_map), 'something is wrong in auc shuffle'


    num_fixations_other = min(ind,num_fixations)

    num_pixels = s_map.shape[0]*s_map.shape[1]
    random_numbers = []
    for i in range(0,splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)	

    aucs = []
	# for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k%s_map.shape[0]-1, k/s_map.shape[0]])
		# in these values, we need to find thresholds and calculate auc
        thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        r_sal_map = np.array(r_sal_map)

		# once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0,0.0))
        for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map>=thresh] = 1.0
            num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
            tp = num_overlap/(num_fixations*1.0)
			
			#fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

            area.append((round(tp,4),round(fp,4)))
		
        area.append((1.0,1.0))
        area.sort(key = lambda x:x[0])
        tp_list =  [x[0] for x in area]
        fp_list =  [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))
	
    return np.mean(aucs)



def nss(s_map,gt):
    """
    Normalized scan path saliency
    """
	#gt = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

    x,y = np.where(gt==1)
    temp = []
    for i in zip(x,y):
        temp.append(s_map_norm[i[0],i[1]])
    return np.mean(temp)


def infogain(s_map,gt,baseline_map):
    """
    Information gain
    """
	#gt = discretize_gt(gt)
	# assuming s_map and baseline_map are normalized
    eps = 2.2204e-16

    s_map = s_map/(np.sum(s_map)*1.0)
    baseline_map = baseline_map/(np.sum(baseline_map)*1.0)

	# for all places where gt=1, calculate info gain
    temp = []
    x,y = np.where(gt==1)
    for i in zip(x,y):
        temp.append(np.log2(eps + s_map[i[0],i[1]]) - np.log2(eps + baseline_map[i[0],i[1]]))

    return np.mean(temp)



def similarity(s_map,gt):
    """
    Similarity
    """
	# here gt is not discretized nor normalized
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    s_map = s_map/(np.sum(s_map)*1.0)
    gt = gt/(np.sum(gt)*1.0)
    x,y = np.where(gt>0)
    sim = 0.0
    for i in zip(x,y):
        sim = sim + min(gt[i[0],i[1]],s_map[i[0],i[1]])
    return sim


def cc(s_map,gt):
    """
    Pearson's correlation coefficient
    """
    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
    gt_norm = (gt - np.mean(gt))/np.std(gt)
    a = s_map_norm
    b= gt_norm
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def kldiv(s_map,gt):
    """
    Saliency metric based on Kullback-Leibler divergence
    """
    s_map = s_map/(np.sum(s_map)*1.0)
    gt = gt/(np.sum(gt)*1.0)
    eps = 2.2204e-16
    return np.sum(gt * np.log(eps + gt/(s_map + eps)))

