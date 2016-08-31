# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:53:23 2016
functions for making histogram of data
@author: ttruszko
"""

import h5py
import pandas as pd

#f.close()
#drive_path='d:/
def hist_single_cell(cell_specimen_id, drive_path, letter):
    f, path = open_h5_file(cell_specimen_id, drive_path, letter)
    mean_sweep_response=pd.read_hdf(path, 'analysis/mean_sweep_response_ns')
    #collect mean sweep response
    #    response = f['analysis']['response_ns'].value

    
def open_h5_file(cell_specimen_id, drive_path, letter):
    exp_container_id = get_container_id(cell_specimen_id)
    session_id = get_session_id(exp_container_id, letter)
    path = drive_path + '/BrainObservatory/ophys_analysis/' + str(session_id) + '_three_session_B_analysis.h5'
    f = h5py.File(path, 'r')
    response = f['analysis']['response_ns'].value
    f.close()
    mean_sweep_response=pd.read_hdf(path, 'analysis/mean_sweep_response_ns')
    sweep_response = pd.read_hdf(path, 'analysis/sweep_response_ns')
    return(response, mean_sweep_response, sweep_response)
    
    

#print(open_h5_file(cell_specimen_id, 'd:', 'B'))    
print(hist_single_cell(cell_specimen_id, 'd:', 'B'))    
#    response = f['analysis']['response_ns'].value
#    f.close()
#
#sweep_response = pd.read_hdf(path, 'analysis/sweep_response_ns')
#mean_sweep_response=pd.read_hdf(path, 'analysis/mean_sweep_response_ns')