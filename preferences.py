#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:34:49 2020

@author: zeynep
"""

"""
I save some variables which take long time to compute as pkl files, and carry on
the processing in the subsequent files by simply loading these variables. However,
if you want to compute from scratch, then you may set the FROM_SCRATCH ket to TRUE.

Similarly, in some cases I keep track of the intermediate results by displaying
images, graphs etc. If youwant to see these, you may set DISPLAY_PROCESSING to 
TRUE.
"""
FROM_SCRATCH = False
DISPLAY_PROCESSING = False

"""
SCALE_PERCENT is for displaying large images, especially many of them on one 
screen. It is between 0 and 1.
"""
SCALE_PERCENT = 0.30 

"""
 considering the sampling rate of the gaze tracker (90hz)
 and duration of the target image display (about 1 sec)
 I require at least 90 samples and at most 100 samples to consider that 
 time window
 
 Note that target image is display for long time (2.5 sec) before any image is
 displayed. This gives experimenter some time to walk away from the computer
 
 So the NMAX_FIXATIONS variable solves that problem too. That is, I only
 want to consider the targets between object images
 """
NMIN_FIXATIONS = 80
NMAX_FIXATIONS = 100

PERSON_MIN_AGE = 30
PERSON_MAX_AGE = 100

"""
When I sample from the empirical distributions, I sample N_RANDOM_SAMPLES many
data points. 

NBINS_LOG is used while building an empirical histogram of the log of the 
displacement. 

"""
N_RANDOM_SAMPLES = 100
NBINS_LOG = 30 # number of bins in histogram of log of dist

"""
As baseline saliency, in the manuscript we report only spectral residual. But 
actually we tried with 3 baseline saliency methods. However, the fine grain and 
objectness methods did not perform good enought so we decided to use the best 
of the three, that is spectral resiudal. Nevertheless, for the inertested we 
keep the following in the array. Please set the desired baseline method by 
chaging the key to True. 
"""
SALIENCIES = {
'STATIC_SPECRES' : True,\
'FINE_GRAIN' : False,\
'OBJECTNESS' : False\
}
    

    
"""
Standard saliency metrics. Please change the metric of interest by setting the 
keys True/False
 'AUC_JUDD': False,\
 'AUC_BORJI': False,\
 'AUC_SHUFF': False,\
 'NSS': True,\
 'INFOGAIN': True,\
 'SIM': True,\
 'CC': True,\
 'KLDIV': True,\
 I usually like the below
  'NSS',\
 'INFOGAIN',\
 'SIM',\
 'CC',\
 'KLDIV',\
"""
SALIENCY_METRICS = \
{
 'AUC_JUDD',\
 'NSS',\
 'INFOGAIN',\
 'SIM',\
 'CC',\
 'KLDIV'\
 }
    
BASELINE_SALIENCY_TYPE = 'STATIC_SPECRES'



DUR_IMAGE = 2500
DUR_DELTA_TAU = 500
DUR_WINDOW = 1000

# for shifting time window over fixations
NSAMPLES_WINDOW = DUR_WINDOW / 1000 * 90 # 400msec * 90 Hz
NSAMPLES_DELTA_TAU = DUR_DELTA_TAU / 1000 * 90# 200 msec * 90 Hz, shifting window by delta_tau = 200msec at every step
NSAMPLES_IMAGE = int( DUR_IMAGE / 1000 * 90)

NWINDOWS = int( (NSAMPLES_IMAGE - NSAMPLES_WINDOW/2 ) / NSAMPLES_DELTA_TAU ) 

KMAX = 10 # max number of clusters (ie saccades)
KOPT_MAX_CLUSTERS = 10 # max number of cluster center candidates
N_OVERSAMPLE = 1000# rate of oversampling for estimating cluster center candidates


"""
When I generate a set of fixations over an image, I need to draw n_saccades and 
n_fixations randomly. Ideally the number of total saccades need to be 250. Thus,
I sample N_SAMPLING_MAX many times and pick the values of n_saccades and 
n_fixations, which give the closest number of samples to 250.

"""
N_SAMPLING_MAX = 50

"""
Since each object image is displayed 2.5 sec and sampling rate is 100, I set 
N_FIXATIONS_RANDOM to 250
"""
N_FIXATIONS_RANDOM = 250 

