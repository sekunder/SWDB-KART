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
#for each area:
#    get exp containers
#    for each exp container:
#        get session B id
#        for each session B id:
#            pull out mean_sweep_response for all cells
#            average each cell over all presentations of an image
#            average all cells over each image
#            save average_expt cont id
#    in each area, average the entire pop response to generate global avg
#    in each area, plot responses (offset)
    
# In[]
# general initialization
from data_preprocessing import BOC_init
boc, Selectivity_S_df, y = BOC_init()    
import pandas as pd
# In[]
# generate and save figs
targeted_structures = boc.get_all_targeted_structures()    
expt_cont_list = pd.DataFrame(boc.get_experiment_containers())
for struct in targeted_structures:
    expt_cont_VISlist=expt_cont_list[expt_cont_list['targeted_structure']==struct]
    expt_cont_ids=expt_cont_VISlist['id'].unique()
    for cont_id in expt_cont_ids:
        session_id, session_data=get_session_id(cont_id, 'B')
        print(session_id)
        #exp_cont_id=expt_cont_ids[expt_cont['id']]
    #print(expt_cont_ids)
        

    
# In[]
    #testing area
#boc.get_experiment_containers?
    
# In[]
#do some statistics