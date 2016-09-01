# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:12:32 2016

@author: ttruszko

This script will generate the figure needed for generating the population response figure.

figure=average response of a whole experiment container to each image.
Repeat for each area.
Quantify how??
generate average pop response for overlay.
"""

# In[]
# general initialization
from data_preprocessing import * 
boc, Selectivity_S_df, y = BOC_init(selectivity_csv_filename='C:\\Users\\ttruszko\\Documents\\GitHub\\SWDB-KART\\image_selectivity_dataframe.csv')    
import pandas as pd

# In[]
# new function to get h5 file data
def open_h5_file_sessionID(session_id, drive_path, letter):
    #session_id, session_data = get_session_id(exp_container_id, letter)
    path = drive_path + '/BrainObservatory/ophys_analysis/' + str(session_id) + '_three_session_B_analysis.h5'
    print(path)    
    f = h5py.File(path, 'r')
    response = f['analysis']['response_ns'].value
    f.close()
    mean_sweep_response=pd.read_hdf(path, 'analysis/mean_sweep_response_ns')
    sweep_response = pd.read_hdf(path, 'analysis/sweep_response_ns')
    stim_table_ns = pd.read_hdf(path, 'analysis/stim_table_ns')
    return(response, mean_sweep_response, sweep_response, stim_table_ns)
    
# In[]
# generate and save mean population response for each expt container for each image
    
targeted_structures = boc.get_all_targeted_structures()
#targeted_structure=targeted_structures[3]     
expt_cont_list = pd.DataFrame(boc.get_experiment_containers())
#expt_cont_ids=[511511015, 511510733]
mean_bystructure=[]
for struct in targeted_structures:
    expt_cont_VISlist=expt_cont_list[expt_cont_list['targeted_structure']==struct]
    expt_cont_ids=expt_cont_VISlist['id'].unique()
    mean_by_container=[]    
    for cont_id in expt_cont_ids:
        session_id, session_data=get_session_id(cont_id, 'B', boc=boc)
        response, mean_sweep_response, sweep_response, stim_table_ns = open_h5_file_sessionID(session_id, 'd:', 'B') 
        
        mean_sweep_response_nd = mean_sweep_response.values 
        mean_perstim=[] 
        img_ids = sorted(stim_table_ns['frame'].unique())
        
        for stim_num in img_ids:
            img_idx=stim_table_ns['frame']==stim_num
            all_img_data=mean_sweep_response[img_idx]
            small_mean=all_img_data.mean()
            big_mean=small_mean.mean()
            mean_perstim.append(big_mean)
        print('container id: ', cont_id)
        mean_by_container.append(mean_perstim)    
    mean_bystructure.append(mean_by_container)

# save 
list_of_dicts = [{'area':area, 'list_of_arrays':np.asarray(tuple(A))} for area, A in zip(targeted_structures, mean_bystructure)]
np.save('mean_of_mean_population', list_of_dicts)
# In[]
    #testing area
#boc.get_experiment_containers?
#for A in list_of_arrays:
#    print(A.shape)  

# In[]
#make a plot for each structure

import matplotlib.pyplot as plt

# In[]
#do some statistics