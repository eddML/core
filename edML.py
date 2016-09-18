#!/usr/bin/env python


import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
import scipy.io as sio
import numpy as np 
import load_features
from load_features import get_feature_label
from train_model import train_and_tune_SVC, train_and_tune_RFC, score_train_test
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def border_indices(center_i,center_j,polarity,r):
        x = np.array([])
        y = np.array([])
        if polarity == 1:                                       # CCW
                for i in range(center_i-r, center_i+r+1):
                        j= (r**2 - (i-center_i)**2)**0.5 + center_j
                        y = np.append(y, round(i))
                        x = np.append(x, round(j))

                for i in range(center_i+r, center_i-r-1, -1):
                        j= center_j - (r**2 - (i-center_i)**2)**0.5
                        y = np.append(y, round(i))
                        x = np.append(x, round(j))
        
        elif polarity == -1:                                       # CW
                for i in range(center_i+r, center_i-r-1, -1):
                        j= center_j - (r**2 - (i-center_i)**2)**0.5
                        y = np.append(y, round(i))
                        x = np.append(x, round(j))

                for i in range(center_i-r, center_i+r+1):
                        j= (r**2 - (i-center_i)**2)**0.5 + center_j
                        y = np.append(y, round(i))
                        x = np.append(x, round(j))

        return x, y


def integrate_phase(snapshot, center_i, center_j, r):
        H, W = snapshot.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        d2 = (x - center_j)**2 + (y - center_i)**2
        mask = d2 <= r**2
        all_points = np.sum(mask)
        #phase_sum = np.sum(snapshot[np.where(mask>0)])
        phase_sum = np.sum(  np.sin(snapshot[np.where(mask>0)]*np.pi/180.0)  )
        phase_norm = phase_sum/float(all_points)
        return mask, phase_sum, phase_norm


def get_eddy_radius(snapshot, center_i, center_j,polarity):
	s =np.array([])
	inter =np.array([])
	rv =np.array([])
	rv2 =np.array([])
	pv =np.array([])
	H, W = snapshot.shape

	r = 20
	while (1):
		r += 1
                if r+center_i >= H-4 or center_i-r < 4 or r+center_j >= W-4 or center_j-r < 4:
                        break

		x, y = border_indices(center_i, center_j, polarity, r)
		x = x.astype(int)
		y = y.astype(int)
		phase = snapshot[y,x]
		
		##############  getting radius by a linear fit  ################
		
		index = np.array(range(0, len(phase)))
		slope, intercept, r_value, p_value, std_err = stats.linregress(index,phase)
		s = np.append(s, slope)
		inter = np.append(inter, intercept)
		rv = np.append(rv, r_value)
		rv2 = np.append(rv2, r_value**2)
		pv = np.append(pv, p_value)
		
		if r_value**2 <= 0.80:
			break
		
		################################################################



        x = np.array([])
        y = np.array([])
        for i in range(-2,3):
                a, b = border_indices(center_i, center_j, 1, r+i)
                a = a.astype(int)
                a = a.astype(int)
                x = np.append(x,a)
                y = np.append(y,b)
        x = x.astype(int)
        y = y.astype(int)

	return r, y, x, r_value**2, p_value



def get_eddy_domain(snapshot, center_i, center_j):
	H, W = snapshot.shape
        r = 20
        while (1):
                r += 1
                if r+center_i >= H-4 or center_i-r < 4 or r+center_j >= W-4 or center_j-r < 4:
                        break 

                x, y = border_indices(center_i, center_j, 1, r)
                x = x.astype(int)
                y = y.astype(int)

		mask, phase_sum, phase_norm = integrate_phase(snapshot, center_i, center_j, r)	
		oscillators = np.sum(mask)
                if abs(phase_sum) >= 3.5*oscillators:
                        break

	x = np.array([])
	y = np.array([])
	for i in range(-1,2):
	        a, b = border_indices(center_i, center_j, 1, r+i)
        	a = a.astype(int)
	        a = a.astype(int)
		x = np.append(x,a)
		y = np.append(y,b)
	x = x.astype(int)
	y = y.astype(int)
	return r, mask, y, x, phase_sum, phase_norm	


def get_sla(itnum):
        dataset = sio.loadmat('provide file path' % itnum)
        ssh = dataset['ssh']
        col = dataset['col']
        row = dataset['row']
        ssh = ssh.reshape(row[0], col[0])
        ssh[np.isnan(ssh)] = np.nanmean(ssh)
        return ssh


def get_vort(itnum):
        dataset = sio.loadmat('provide file path' % itnum)
        vort = dataset['vort']
        col = dataset['col']
        row = dataset['row']
        vort = vort.reshape(row[0], col[0])
        vort[np.isnan(vort)] = np.nanmean(vort)
        return vort


def get_mask():
	# provide your own mask fuction for the region of your interest
	# the function returns the location indecies and the correspondin lat/lon of the land data points	
        return cols, rows, mask_lon[cols], mask_lat[rows]










itnum_start = int(sys.argv[1])
itnum_end = int(sys.argv[2])

f_width = load_features.f_width
f_height = load_features.f_height
samples = load_features.samples

#########################  Load Features  #########################
features, eddy, polarity = get_feature_label(2)       # X: feature matrix      y1: eddy vector       y2: polarity vertor 
###################################################################


##########################  Classifiers  ##########################
# use the helper functions in the train_model.py file to train and validate and evaluate
### load trained classifieres
clf = pickle.load(open('clf_core.pck', 'rb'))
clf_polarity = pickle.load(open('clf_pol.pck', 'rb'))




######################  Load Vphase Snapshot  ######################
for itnum in range(itnum_start,itnum_end+1):
  tic = time.clock()
  vphase_path = 'provide file path'
  dataset_2pi = sio.loadmat('%svphase_2pi_%10.10d.mat' % (vphase_path,itnum))
  #dataset_pi2 = sio.loadmat('%svphase_pi2_%10.10d.mat' % (vphase_path,itnum))
  #dataset_pi = sio.loadmat('%svphase_pi_%10.10d.mat' % (vphase_path,itnum))
  #dataset_pi = dataset_2pi
  vphase_2pi = dataset_2pi['phase']
  #vphase_pi2 = dataset_pi2['phase']
  #vphase_pi = dataset_pi['phase']
  snapshot = vphase_2pi
  snapshot = np.nan_to_num(snapshot)
  marked_snapshot = snapshot
  eddy_cores = 0 * snapshot
  #snapshot_pi = vphase_pi
  #snapshot_pi = np.nan_to_num(snapshot_pi)
  marked_sla = get_sla(itnum)
  #marked_vort = get_vort(itnum)
  radius = int(f_width/2)
  height = snapshot.shape[0] /1
  width = snapshot.shape[1] /1




  identified_eddies = 0
  identified_ccw = 0
  identified_cw = 0
  eddy_centers_i = np.array([])
  eddy_centers_j = np.array([])
  eddy_polarity = np.array([])
  eddy_radius = np.array([])
  radius_rv2 = np.array([])
  radius_pv = np.array([])
  for center_i in range(radius , height-radius):
    for center_j in range(radius , width-radius):
      try:
        if eddy_cores[center_i, center_j] <> 0:
          continue
        snapshot_cut = snapshot[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1]
        if clf.predict(snapshot_cut.reshape(f_width*f_height))[0] == 1:			###  Identify Eddy
	  pred_pol = clf_polarity.predict(snapshot_cut.reshape(f_width*f_height))[0]	###  Identify Polarity
          try:
              ###################  settining eddy radius using phase on the border  #######################
              edd_radius, indices_i, indices_j, rv2, pv = get_eddy_radius(snapshot, center_i, center_j, pred_pol)
	      if edd_radius < 25:		#####  min radius
		continue
              identified_eddies += 1
              eddy_cores[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1] = 1
              eddy_centers_i = np.append(eddy_centers_i , center_i)
              eddy_centers_j = np.append(eddy_centers_j , center_j)
	      eddy_polarity = np.append(eddy_polarity , pred_pol)	
              eddy_radius = np.append(eddy_radius , edd_radius)
              radius_rv2 = np.append(radius_rv2 , rv2)
              radius_pv = np.append(radius_pv , pv)
	      if pred_pol == 1:	
                identified_ccw += 1
                marked_snapshot[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1] = -1
                marked_sla[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1] = 30 
                marked_snapshot[indices_i, indices_j] = -1
                marked_sla[indices_i, indices_j] = 30 
	      else:
                identified_cw += 1
                marked_snapshot[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1] = 370
                marked_sla[center_i-radius:center_i+radius+1 , center_j-radius:center_j+radius+1] = -30
                marked_snapshot[indices_i, indices_j] = 370
                marked_sla[indices_i, indices_j] = -30
              #############################################################################################
          except Exception as e:
	      continue
              print('Exception: ' , str(e))
      except Exception as e:
        print('Error:  ' , str(e))
  toc = time.clock()
  print('process time: ', toc-tic)
  store_path = 'provide file path'
  np.savez(store_path % itnum , identified_eddies=identified_eddies , identified_ccw=identified_ccw, identified_cw=identified_cw , eddy_centers_i=eddy_centers_i , eddy_centers_j=eddy_centers_j , eddy_polarity=eddy_polarity , eddy_radius=eddy_radius , radius_rv2=radius_rv2 , radius_pv=radius_pv)

  print('----------------------------')
  print('itnum: ',itnum)
  print('Identified Eddied: ',identified_eddies)
  print('Identified CCW: ', identified_ccw)
  print('Identified CW: ', identified_cw)
  print('----------------------------')
  print('')





  im = plt.imshow(marked_snapshot)
  plt.gca().invert_yaxis()
  plt.title(str(itnum))
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("right", size="3%", pad=0.05)  
  plt.colorbar(im, cax=cax)
  store_path = 'provide file path'
  plt.savefig(store_path % itnum , bbox_inches='tight' , dpi=300)
  #plt.show(block=True)
  plt.close()


  cols, rows, mask_lon, mask_lat = get_mask()   ## provide a land mask (if applicable)


  plt.figure()
  f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
  im = ax1.imshow(marked_sla, cmap='RdBu', vmin=-0.18, vmax=0.18, extent=[188,209,29,17])                
  ax1.plot(mask_lon, mask_lat, '.', color=(0.3,0.3,0.3), markersize=2)
  ax1.text(188.6,25.7,'a', fontsize=15, fontweight='bold')
  ax1.set_xlim([188, 209])
  ax1.set_ylim([15, 29])
  ax1.set_ylabel('Latitude')
  ax1. set_title(str(itnum)+'\nSea Level Anomaly (cm)')	
  f.colorbar(im, ax=ax1)

  im = ax2.imshow(marked_snapshot, cmap='jet', vmin=0, vmax=360, extent=[188,209,29,17])
  ax2.plot(mask_lon, mask_lat, '.', color=(0.7,0.7,0.7), markersize=2)
  ax2.text(188.6,25.7,'b', fontsize=15, fontweight='bold')
  ax2.set_xlim([188, 209])
  ax2.set_ylim([15, 29])
  ax2.set_ylabel('Latitude')
  ax2.set_xlabel('Longitude')
  ax2.set_title('Surface Velocity Phase (deg)')
  #divider = make_axes_locatable(ax2)
  #cax = divider.append_axes("right", size="3%", pad=0.05)
  #f.colorbar(im, cax=cax)
  f.colorbar(im, ax=ax2)
  store_path = 'provide file path'
  plt.savefig(store_path % itnum , dpi=300)
  #plt.show(block=True)
  plt.close()
####################################################################






