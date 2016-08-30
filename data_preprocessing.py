# WARNING! Be sure to change the line that reads the .csv file, below, should be line 47
# If it doesn't work, you'll need to make the path point to your copy of that file (which should be in this folder)

# We need to import these modules to get started
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

# The dynamic brain database is on an external hard drive.
#drive_path = '/Volumes/Brain2016'
drive_path= 'd:'
manifest_path = os.path.join(drive_path, 'BrainObservatory/manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)
expt_containers = boc.get_experiment_containers()

# Things I will want eventually:
# 1. A dataframe with columns for...
#	a. all the identifiers from the brain observatory
#	b. the preferred directions from the grating and place field stimuli
# 2. A numpy array which is N_neurons x N_timesamples, where entry i,j is 1 or 0 based on whether
#	 the neuron was "firing" at that point (how we decide that we'll get to later

# Data frames first:
# expt_containers_df = pd.DataFrame(expt_containers)

# The stimuli I'm interested in are basically everything except the natural movie and natural scenes
non_movie_stimuli = ['drifting_gratings', 'locally_sparse_noise', 'spontaneous', 'static_gratings']

all_expts_df = pd.DataFrame(boc.get_ophys_experiments(stimuli=non_movie_stimuli))
# this has headers:
# age_days	cre_line	experiment_container_id	id	imaging_depth	session_type	targeted_structure

# seems like I can use get_cell_speciments to get everything I'm after
specimens_df = pd.DataFrame(boc.get_cell_specimens(experiment_container_ids=all_expts_df.experiment_container_id.values))
# this has headers:
# area	cell_specimen_id	dsi_dg	experiment_container_id	imaging_depth	osi_dg	osi_sg	p_dg	p_ns	p_sg
# pref_dir_dg	pref_image_ns	pref_ori_sg	pref_phase_sg	pref_sf_sg	pref_tf_dg	time_to_peak_ns	time_to_peak_sg
# tld1_id	tld1_name	tld2_id	tld2_name	tlr1_id	tlr1_name

# There's also a handy bit of data from Saskia, in the form of a measurement called S. See
# Decoding Visual Inputs From Multiple Neurons in the Human Temporal Lobe, J. Neurophys 2007, by Quiroga et al


selectivity_S_df = pd.read_csv(os.path.join('image_selectivity_dataframe.csv'), index_col=0)
# selectivity_S_df.head()
selectivity_S_df = selectivity_S_df[['cell_specimen_id', 'selectivity_ns']]

specimens_with_selectivity_S = specimens_df.merge(selectivity_S_df,how='outer', on='cell_specimen_id')

# This is all cells in VISp that have a value for the specified parameters (i.e not NaN)
discard_nan = ['selectivity_ns','osi_sg','osi_dg','time_to_peak_ns','time_to_peak_sg']
VISp_cells_with_numbers = specimens_with_selectivity_S[specimens_with_selectivity_S.area == 'VISp']
for col_name in discard_nan:
	VISp_cells_with_numbers = VISp_cells_with_numbers[np.abs(VISp_cells_with_numbers[col_name]) >= 0]
