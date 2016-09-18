from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle

#itnums = []
#itnums = np.append(itnums, range( '<provide the list of files containing the eddy cores>' ))
#.
#.
#.



global running_edd
global dead_edd
global lost 

running_edd = []       ### eddy record struc: [ Birth:<integer>, Polarity:<integer>, LastSeen:<integer>, Trajectory:<[time, loc_row, loc_col]>, Radius:<[time, Radius, R-Value2, p-Value]>]
dead_edd = []
res = 0.01
lost = 2               ### if an eddy core is not identified more than "lost", it is assumed dead
min_lifetime = 4       ### detected eddies with lifetimes smaller than this value will be rejected

travel_range = 50      ### maximum displacement of an eddy between two subsequent identification of eddy cores. 
		       ### it is based on index change (distance = travel_range*resolution(deg) )



def load_cores(itnum):
	path = 'provide file path'
	data = np.load(path % itnum)
	#eddies = eddies + data['identified_eddies']
	#ccw = ccw + data['identified_ccw']
	#cw = cw + data['identified_cw']
	eddy_centers_i = data['eddy_centers_i']
        eddy_centers_j = data['eddy_centers_j']
        eddy_polarity = data['eddy_polarity']
	eddy_radius = data['eddy_radius']
	radius_rv2 = data['radius_rv2']
	radius_pv = data['radius_pv']
	return eddy_centers_i, eddy_centers_j, eddy_polarity, eddy_radius, radius_rv2, radius_pv


def cleanup(current_itnum):
	running_edd_copy = list(running_edd)
	for i in range(0,len(running_edd_copy)):
		rec = running_edd_copy[i]
		if current_itnum-rec[2] > lost:             
			dead_edd.append(rec)
			running_edd.remove(rec)	



def create_eddy_record(itnum, polarity, center_i, center_j, radius, rv2, pv):
	rec = [itnum, polarity, itnum, [[itnum, center_i, center_j]], [[itnum, radius, rv2, pv]] ]
	return rec


def find_eddy(pol, cent_i, cent_j):
	for i in range(0,len(running_edd)):
		fin = len(running_edd[i][3]) - 1
		if pol==running_edd[i][1] and abs(running_edd[i][3][fin][1]-cent_i)<travel_range and abs(running_edd[i][3][fin][2]-cent_j)<travel_range:
			return i
	return -1		  


def within_margin(i, j, margin, up, right):
        within = False
        if (i - margin) * (up - i) <= 0:
                within = True
        if (j - margin) * (right - j) <= 0:
                within = True
        return within



def load_selected_traj_indices():
        indices = np.array([])
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]:
                text_file = open('provide file path', "r")
                lines = text_file.readlines()
                for j in range(0, len(lines)):
                        indices = np.append(indices, int(lines[j]))
                text_file.close()

        return indices


cent_i, cent_j, pol, rad, rv2, pv = load_cores(itnums[0])
for i in range(0,len(pol)):
	running_edd.append(create_eddy_record(itnums[0], pol[i], cent_i[i], cent_j[i], rad[i], rv2[i], pv[i])) 

for itnum in itnums[1:]:  
	print('Loading: '+str(itnum), end='\r')
	cleanup(itnum)
	cent_i, cent_j, pol, rad, rv2, pv = load_cores(itnum)
	for i in range(0,len(pol)):
		edd_index = find_eddy(pol[i], cent_i[i], cent_j[i])	
		if edd_index<0:
			running_edd.append(create_eddy_record(itnum, pol[i], cent_i[i], cent_j[i], rad[i], rv2[i], pv[i]))		### new eddy
		else:
			running_edd[edd_index][2]=itnum				### LastSeen update
			temp=running_edd[edd_index][3]
			temp.append([itnum, cent_i[i], cent_j[i]])
			running_edd[edd_index][3]=temp				### trajectory update
                        temp1=running_edd[edd_index][4]
                        temp1.append([itnum, rad[i], rv2[i], pv[i]])
                        running_edd[edd_index][4]=temp1                         ### radius update

for i in range(0,len(running_edd)):
	dead_edd.append(running_edd[i])

###################### Reject Short-Lifetime Eddies #####################
dead_edd_copy = list(dead_edd)
for i in range(0,len(dead_edd)):
	tau = dead_edd[i][2]-dead_edd[i][0]+1
	if tau < min_lifetime:
		dead_edd_copy.remove(dead_edd[i])		
dead_edd = list(dead_edd_copy)





#########################################################################
print('Number of Tracks: ', str(len(dead_edd)))
outfile = open('eddy_archive.pck', 'w')
pickle.dump(dead_edd, outfile)
outfile.close()



'''
################### Only Select the 'Selected' Trajectories  #######################
selected_dead_edd = []
indices = load_selected_traj_indices()
for index in list(indices):
        selected_dead_edd.append(dead_edd[int(index)])
dead_edd = list(selected_dead_edd)
####################################################################################
print('LENGTH::::: ', str(len(dead_edd)))
'''

lifetime_ccw = []
lifetime_cw = []
displacement_ccw = []
displacement_cw = []
mean_vel_ccw = []
mean_vel_cw = []
dir_ccw = []
dir_cw = []
radius_ccw = []
radius_cw = []


for i in range(0,len(dead_edd)):
	if dead_edd[i][1]==1:
		tau = dead_edd[i][2]-dead_edd[i][0]+1
		fin = len(dead_edd[i][3])-1
		delta_i = dead_edd[i][3][fin][1] - dead_edd[i][3][0][1]
		delta_j = dead_edd[i][3][fin][2] - dead_edd[i][3][0][2]
		delta_x = 111 * 0.01 * delta_i
		delta_y = 111 * 0.01 * delta_j
		displacement = (delta_x**2 + delta_y**2)**0.5
		v_i = delta_x/tau
		v_j = delta_j/tau
		v = (v_i**2 + v_j**2)**0.5
		lifetime_ccw.append(tau)
		displacement_ccw.append(displacement)
		mean_vel_ccw.append(v)
		dir_ccw.append(np.arctan2(delta_i,delta_j)*180/3.14)	
		for j in range(0, len(dead_edd[i][4])):
			radius_ccw.append(dead_edd[i][4][j][1])
        if dead_edd[i][1]==-1:
                tau = dead_edd[i][2]-dead_edd[i][0]+1
                fin = len(dead_edd[i][3])-1
                delta_i = dead_edd[i][3][fin][1] - dead_edd[i][3][0][1]
                delta_j = dead_edd[i][3][fin][2] - dead_edd[i][3][0][2]
                delta_x = 111 * 0.01 * delta_i
                delta_y = 111 * 0.01 * delta_j
                displacement = (delta_x**2 + delta_y**2)**0.5
                v_i = delta_x/tau
                v_j = delta_j/tau
                v = (v_i**2 + v_j**2)**0.5
                lifetime_cw.append(dead_edd[i][2]-dead_edd[i][0]+1)
		displacement_cw.append(displacement)
		mean_vel_cw.append(v)
		dir_cw.append(np.arctan2(delta_i,delta_j)*180/3.14)
                for j in range(0, len(dead_edd[i][4])):
                        radius_cw.append(dead_edd[i][4][j][1])



lifetime_ccw = np.array(lifetime_ccw)
lifetime_cw = np.array(lifetime_cw)
displacement_ccw = np.array(displacement_ccw)
displacement_cw = np.array(displacement_cw)
mean_vel_ccw = np.array(mean_vel_ccw)
mean_vel_cw = np.array(mean_vel_cw)
dir_ccw = np.array(dir_ccw)
dir_cw = np.array(dir_cw)
radius_ccw = np.array(radius_ccw)
radius_cw = np.array(radius_cw)


lifetime_total = np.append(lifetime_ccw,lifetime_cw)
displacement_total = np.append(displacement_ccw,displacement_cw)
mean_vel_total = np.append(mean_vel_ccw,mean_vel_cw)
radius_total = np.append(radius_ccw, radius_cw)

print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
print('')
print('Number of Identified CCW Eddied: ', len(lifetime_ccw))
print('Number of Identified CW Eddied: ', len(lifetime_cw))
print('')
print('************  ALL  Eddies  *************')
print('Mean Lifetime of All Eddies: ', np.mean(lifetime_total), 'day')
print('STD Lifetime of All Eddies: ', np.std(lifetime_total), 'day')
print('Mean Displacement of All Eddies: ', np.mean(displacement_total), 'km')
print('Maximum Displacement of All Eddies: ', np.amax(displacement_total), 'km')
print('Minimum Displacement of All Eddies: ', np.amin(displacement_total), 'km')
print('Minimum Lifetime of All Eddies: ', np.amin(lifetime_total), 'day')
print('Maximum Lifetime of All Eddies: ', np.amax(lifetime_total), 'day')
print('Mean Guide Velocity of All Eddies: ', np.mean(mean_vel_total), 'km/day')
print('Minimum Guide Velocity of All Eddies: ', np.amin(mean_vel_total), 'km/day')
print('Maximum Guide Velocity of All Eddies: ', np.amax(mean_vel_total), 'km/day')
print('Mean Radius of All Eddies: ', np.mean(radius_total), 'km')
print('Minimum Radius of All Eddies: ', np.amin(radius_total), 'km')
print('Maximum Radius of All Eddies: ', np.amax(radius_total), 'km')
print('')
print('************  CCW  Eddies  *************')
print('Mean Lifetime of CCW Eddies: ', np.mean(lifetime_ccw), 'day')
print('STD Lifetime of CCW Eddies: ', np.std(lifetime_ccw), 'day')
print('Mean Direction of CCW Eddies: ', np.mean(dir_ccw), 'deg')
print('Mean Displacement of CCW Eddies: ', np.mean(displacement_ccw), 'km')
print('Maximum Displacement of CCW Eddies: ', np.amax(displacement_ccw), 'km')
print('Minimum Displacement of CCW Eddies: ', np.amin(displacement_ccw), 'km')
print('Minimum Lifetime of CCW Eddies: ', np.amin(lifetime_ccw), 'day')
print('Maximum Lifetime of CCW Eddies: ', np.amax(lifetime_ccw), 'day')
print('Mean Guide Velocity of CCW Eddies: ', np.mean(mean_vel_ccw), 'km/day')
print('Minimum Guide Velocity of CCW Eddies: ', np.amin(mean_vel_ccw), 'km/day')
print('Maximum Guide Velocity of CCW Eddies: ', np.amax(mean_vel_ccw), 'km/day')
print('Mean Radius of CCW Eddies: ', np.mean(radius_ccw), 'km')
print('Minimum Radius of CCW Eddies: ', np.amin(radius_ccw), 'km')
print('Maximum Radius of CCW Eddies: ', np.amax(radius_ccw), 'km')
print('')
print('************  CW  Eddies  *************')
print('Mean Lifetime of CW Eddies: ', np.mean(lifetime_cw), 'day')
print('STD Lifetime of CW Eddies: ', np.std(lifetime_cw), 'day')
print('Mean Direction of CW Eddies: ', np.mean(dir_cw), 'deg')
print('Mean Displacement of CW Eddies: ', np.mean(displacement_cw), 'km')
print('Maximum Displacement of CW Eddies: ', np.amax(displacement_cw), 'km')
print('Minimum Displacement of CW Eddies: ', np.amin(displacement_cw), 'km')
print('Minimum Lifetime of CW Eddies: ', np.amin(lifetime_cw), 'day')
print('Maximum Lifetime of CW Eddies: ', np.amax(lifetime_cw), 'day')
print('Mean Guide Velocity of CW Eddies: ', np.mean(mean_vel_cw), 'km/day')
print('Minimum Guide Velocity of CW Eddies: ', np.amin(mean_vel_cw), 'km/day')
print('Maximum Guide Velocity of CW Eddies: ', np.amax(mean_vel_cw), 'km/day')
print('Mean Radius of CW Eddies: ', np.mean(radius_cw), 'km')
print('Minimum Radius of CW Eddies: ', np.amin(radius_cw), 'km')
print('Maximum Radius of CW Eddies: ', np.amax(radius_cw), 'km')
print('')
print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')








##########################   Velocity/Lifetime/Displacement   ########################
f, axarr = plt.subplots(3,1)

lifetime_total = np.append(lifetime_ccw,lifetime_cw)
bins = np.linspace(0, np.amax(lifetime_total), np.amax(lifetime_total)+1)
axarr[0].hist(lifetime_total, bins, color='blue', edgecolor='none', alpha=0.25)
axarr[0].text(5,3000,'a', fontsize=15, fontweight='bold')
axarr[0].set_xlabel('Lifetime (day)')
axarr[0].set_yscale('log')

displacement_total = np.append(displacement_ccw,displacement_cw)
bins = np.linspace(0, np.amax(displacement_total), 141)
axarr[1].text(20,2000,'b', fontsize=15, fontweight='bold')
axarr[1].hist(displacement_total, bins, color='purple', edgecolor='none', alpha=0.25)
axarr[1].set_xlabel('Displacement (km)')
axarr[1].set_yscale('log')

mean_vel_total = np.append(mean_vel_ccw,mean_vel_cw)
bins = np.linspace(np.amin(mean_vel_total), np.amax(mean_vel_total), np.amax(mean_vel_total)-np.amin(mean_vel_total)+1)
bins = np.linspace(np.amin(mean_vel_total), np.amax(mean_vel_total), 100)
axarr[2].text(0.3,380,'c', fontsize=15, fontweight='bold')
axarr[2].hist(mean_vel_total, bins, color='magenta', edgecolor='none', alpha=0.25)
axarr[2].set_xlabel('Guide Velocity (km/day)')
#axarr[1,0].set_yscale('log')

plt.tight_layout()
plt.show(block=True)
######################################################################################


