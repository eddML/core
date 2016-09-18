#!/usr/bin/env python

import scipy.io as sio
import numpy as np


global f_width
global f_height
global samples

f_width = 21
f_height = 21
samples =356		### hawaii training set

def get_feature_label(phase_range):
	if phase_range == 2:
		dataset_2pi = sio.loadmat('provide file path')		## hawaii_eddy_2pi.mat
		features_2pi = dataset_2pi['features_2pi']
		features_2pi = features_2pi.reshape(f_width*f_height,samples)
		features_2pi = np.transpose(features_2pi)
		X = features_2pi
 	        labels = dataset_2pi['labels']
        	others = dataset_2pi['others']
	elif phase_range == 1:		
		dataset_pi = sio.loadmat('provide file path')
		features_pi = dataset_pi['features_pi']
		features_pi = features_pi.reshape(f_width*f_height,samples)
		features_pi = np.transpose(features_pi)
		X = features_pi
                labels = dataset_pi['labels']
                others = dataset_pi['others']
	elif phase_range == 0.5:
		dataset_pi2 = sio.loadmat('provide file path')
		features_pi2 = dataset_pi2['features_pi2']
		features_pi2 = features_pi2.reshape(f_width*f_height,samples)
		features_pi2 = np.transpose(features_pi2)
		X = features_pi2
                labels = dataset_pi2['labels']
                others = dataset_2pi2['others']

	y1 = labels[0]                    ## eddy label
	y2 = labels[1]                    ## eddy polarity

	return (X, y1, y2)
