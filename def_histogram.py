# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:53:23 2016
functions for making histogram of data
@author: ttruszko
"""

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
drive_path='d:'
manifest_path = os.path.join(drive_path,'BrainObservatory/manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)


def open_h5_file(cell_specimen_id, drive_path, letter):
    exp_container_id = get_container_id(cell_specimen_id)
    session_id, session_data = get_session_id(exp_container_id, letter)
    path = drive_path + '/BrainObservatory/ophys_analysis/' + str(session_id) + '_three_session_B_analysis.h5'
    f = h5py.File(path, 'r')
    response = f['analysis']['response_ns'].value
    f.close()
    mean_sweep_response=pd.read_hdf(path, 'analysis/mean_sweep_response_ns')
    sweep_response = pd.read_hdf(path, 'analysis/sweep_response_ns')
    return(response, mean_sweep_response, sweep_response, exp_container_id, session_id, session_data)

def hist_single_cell(cell_specimen_id, drive_path, letter, bins):
    response, mean_sweep_response, sweep_response, exp_container_id, session_id, session_data = open_h5_file(cell_specimen_id, drive_path, letter)
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_data.id.values[0])
    cell_specimen_ids = data_set.get_cell_specimen_ids()    
    cell_idx=np.where(cell_specimen_ids==cell_specimen_id)[0][0]
    cell_series = mean_sweep_response.iloc[:, cell_idx]     
    plt.hist(cell_series, bins=bins)

# In[]
#this is test code. Should return a histogram that looks like a nice histogram with a long tail.
if __name__ == '__main__' : 
    print(hist_single_cell(517510587, 'd:', 'B', 1000))    