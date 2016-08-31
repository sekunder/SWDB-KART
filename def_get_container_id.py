# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:25:25 2016
functions needed for extracting a particular cell
@author: ttruszko
"""
import pandas as pd
from data_preprocessing import BOC_init

boc, x, y = BOC_init() 
    
def get_container_id(cell_specimen_id):
    """This function takes a cell specimen id and returns the experiment container it is in"""
    selectivity_S_df = x
    cell_record=selectivity_S_df.loc[selectivity_S_df['cell_specimen_id']==cell_specimen_id]
    exp_container_id=cell_record.iloc[0]['experiment_container_id']
    return(exp_container_id)

def get_session_id(exp_container_id, letter):
    sessiontype=['three_session_'+str(letter)]
    session_data=pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=[exp_container_id], session_types=sessiontype))
    session_id=session_data['id'][0]
    return(session_id)

# In[]
#this section is a testing section. for cell_specimen_id  517510587, 
#get_container_id should return 511510855 
#get_session_id should return 510705057 (for session B)  
if __name__ == '__main__' :    
    exp_container_id = get_container_id(517510587)
    print(exp_container_id)
    print(get_session_id(exp_container_id, 'B'))

