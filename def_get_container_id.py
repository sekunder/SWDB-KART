# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:25:25 2016
functions needed for extracting a particular cell
@author: ttruszko
"""
import pandas as pd
import os
import allensdk
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

drive_path='d:'
manifest_path = os.path.join(drive_path,'BrainObservatory/manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)
#session_data=pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=[511510855], session_types=['three_session_B']))
#data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_data.id.values[0])
cell_specimen_ids = data_set.get_cell_specimen_ids()

    
def get_container_id(cell_specimen_id):
    """This function takes a cell specimen id and returns the experiment container it is in"""
    drive_path='d:'
    selectivity_S_df = pd.read_csv(os.path.join(drive_path, 'image_selectivity_dataframe.csv'), index_col=0)
    cell_record=selectivity_S_df.loc[selectivity_S_df['specimen_id']==cell_specimen_id]
    exp_container_id=cell_record.iloc[0]['experiment_container_id']
    return(exp_container_id)

def get_session_id(exp_container_id, letter):
    sessiontype=['three_session_'+str(letter)]
    #exp_cont_id = str(exp_container_id)
    session_data=pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=[exp_container_id], session_types=sessiontype))
    #exp_cont_info = boc.get_experiment_containers('id'=[exp_cont_id])
    session_id=session_data['id'][0]
    return(session_id)

    
if __name__ == '__main__' :    
    print(get_container_id(517510587))
    print(get_session_id(exp_container_id, 'B'))
# this should go elsewhere - like in the script to run the data. 
#selectivity_S_df = pd.read_csv(os.path.join(drive_path, 'image_selectivity_dataframe.csv'), index_col=0)
    
# In[]
#help(boc.get_ophys_experiments)
#boc.get_all_session_types()
