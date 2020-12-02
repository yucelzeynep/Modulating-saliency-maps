#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:22:57 2020

@author: zeynep

This file contains the description of the class MyObject. This data structure 
contains information, which is not specific to participants. Namely, the size of 
the object, foreground, background of the images etc. Perhaps, the most important 
component is the definition of the polygons describing the functional and 
manipulative parts.
"""
import numpy as np

import cv2
from scipy import ndimage

import tools_fig as tools_fig

from importlib import reload
import preferences
reload(preferences)

import constants
reload(constants)


class MyObject():
    """
    MyObject is a collection of information on a *single* object (or equivalently an image).
    It does not depend on participant, motivation etc.
    """
    def __init__(self):
        self.image_fname = [] # jpeg name
        
        self.image_orig = []
        self.image_fg_binary = [] # foreground (after thresholding)
        
        self.object_type = [] # one of the six object types
        
        self.object_long_side = 0
        self.object_short_side = 0
        self.object_size = 0
        
        self.smap = {} # saliency map
        for saliency_type in preferences.SALIENCIES:
            self.smap[saliency_type] = []
        
        self.polygon_functional = [] # handle, grip
        self.polygon_manipulative = [] # end effector
        
    def builder(self, image_path):
        """
        Fill in the variables
        """
        
        self.image_fname = image_path.split('/')[-1]
        self.object_type = image_path.split('/')[-2]
        self.load_image(image_path)
        self.get_fg_binary()                   
        self.set_object_dims()        
        self.build_saliency_map()        
        self.load_polygons()
        
#        # save myobject object
#        fpath = constants.OUTPUT_DIR + 'images/' + self.image_fname.split('.')[0] + '.pkl'
#        with open(str(fpath), 'wb') as f:
#            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def load_image(self, image_path):
        """
        Load the original image
        """
        # second variable is set to 0 for reading at gray level, to 1 for color
        self.image_orig = cv2.imread(image_path, 0)
            
  			
    
    def compute_smap_specres(self):
        """
        Compute the saliency map with spectral residual
        """
        saliency_static_specres = cv2.saliency.StaticSaliencySpectralResidual_create()

        if preferences.SALIENCIES['STATIC_SPECRES']:
            (success, saliencyMap_static_specres) = saliency_static_specres.computeSaliency(\
            self.image_orig)
            saliencyMap_static_specres = (saliencyMap_static_specres * 255).astype("uint8")
            
            self.smap['STATIC_SPECRES'] = saliencyMap_static_specres
      
            tools_fig.scale_and_display(saliencyMap_static_specres, "STATIC_SPECRES")
            
    def compute_smap_finegrain(self):
        """
        Compute the saliency map with fine grain
        """
        saliency_static_finegrain = cv2.saliency.StaticSaliencyFineGrained_create()
        
        if preferences.SALIENCIES['FINE_GRAIN']:
            (success, saliencyMap_static_finegrain) = saliency_static_finegrain.computeSaliency(self.image_orig)
            saliencyMap_static_finegrain = (saliencyMap_static_finegrain * 255).astype("uint8")
            
            self.smap['FINE_GRAIN'] = saliencyMap_static_finegrain                       
            
            tools_fig.scale_and_display(saliencyMap_static_finegrain, "FINE_GRAIN")
            
    def compute_smap_objectness(self):
        """
        Compute the saliency map with objectness
        """
        # For this data set, max object detections per image is 1
        MAX_DETECTIONS = 1 
        
        output = []
        saliency_objnes = cv2.saliency.ObjectnessBING_create()
        saliency_objnes.setTrainingPath('objectness_trained_model/')
            
        if preferences.SALIENCIES['OBJECTNESS']:                    
            (success, saliencyMap_objness) = saliency_objnes.computeSaliency( self.image_orig )
            numDetections = saliencyMap_objness.shape[0]
        
            # loop over the detections (redundant)
            for i in range(0, min(numDetections, MAX_DETECTIONS)):
                	# extract the bounding box coordinates
                	(startX, startY, endX, endY) = saliencyMap_objness[i].flatten()
                	
                	# randomly generate a color for the object and draw it on the image
                	output = self.image_orig.copy()
                	color = np.random.randint(0, 255, size=(3,))
                	color = [int(c) for c in color]
                	cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
                
            self.smap['OBJECTNESS'] = saliency_objnes
                
            tools_fig.scale_and_display(output, 'OBJECTNESS')
            
    def build_saliency_map(self):
        """
        Compute saliency maps according to different paradigms
        Check preferences for the keys to control types of saliency
        """
        self.compute_smap_specres()
        self.compute_smap_finegrain()
        self.compute_smap_objectness()
            
    def get_fg_binary(self):
        """
        the image displayed to the participant is gray scale. but there are 3 
        channels
        """
        
        fg_image = self.image_orig.copy()
        
        """
        No need for conversion to gray scale, if you read the image already as 
        gray scale.Use the following to read directly at gray scale 
        cv2.imread(xxx, 0)
         
        Otherwise, threshold using
        fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
        """

        # thresholding
        ret3,fg_binary = cv2.threshold(fg_image, 0, 255, \
                                       cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) # cv2.THRESH_BINARY+

        # morphological operations
        kernel = np.ones((2,2),np.uint8)
        fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
        
        self.image_fg_binary = fg_binary.copy()

       
        
    def set_object_dims(self):
        """
        Set size (number of white pixels on fg image) and 
        long and short sides of the object
        """
        self.object_size = int(np.sum(self.image_fg_binary)/255)
            
    def load_polygons(self):
        """
        Load annotated polygons for functional and manipulative object parts
        """
        # functional
        poly_func_fname = constants.ANNOTATION_POLYGON_FUNCTIONAL_DIR + self.image_fname.replace('.jpeg', '_halfsize.txt')
        
        # polygons are annotated on half-size images so scale back
        poly_func = np.multiply( \
                                np.loadtxt(poly_func_fname, \
                                           dtype='i', delimiter=' '), 2)
        
        self.polygon_functional = poly_func
                
        # manipulative
        poly_manip_fname = constants.ANNOTATION_POLYGON_MANIPULATIVE_DIR + self.image_fname.replace('.jpeg', '_halfsize.txt')
	# polygons are annotated on half-size images so scale back
        poly_manip =  np.multiply( \
                                  np.loadtxt(\
                                             poly_manip_fname, \
                                             dtype='i', delimiter=' '), 2)
        
        self.polygon_manipulative = poly_manip
            
               
            
            
            
        
        
        
        
        
