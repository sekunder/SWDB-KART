# WARNING! Be sure to change the line that reads the .csv file, below, should be line 47
# If it doesn't work, you'll need to make the path point to your copy of that file (which should be in this folder)

# We need to import these modules to get started
import numpy as np
import pandas as pd
import os
import sys
import h5py
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


# The dynamic brain database is on an external hard drive.
# drive_path = '/Volumes/Brain2016'
# if sys.platform.startswith('w'):
# 	drive_path= 'd:'

def BOC_init(stimuli_to_use={
	'drifting_gratings',
	'locally_sparse_noise',
	'spontaneous',
	'static_gratings'
},
		selectivity_csv_filename='image_selectivity_dataframe.csv',
		areas={
			'VISp'
		},
		discard_nan={
			'selectivity_ns',
			'osi_sg',
			'osi_dg',
			'time_to_peak_ns',
			'time_to_peak_sg',
		}
):
	""""Returns a BrainObservatoryCache initialized using the data on the external harddrives provided by the Allen Institute
Example usage:
	boc, specimens_with_selectivity_S, VISp_cells_with_numbers = BOC_init()

Input:
stimuli_stimuli_to_use : Which stimuli we are interested in. By default, this is
	['drifting_gratings', 'locally_sparse_noise', 'spontaneous', 'static_gratings']

selectivity_csv_filename : The path (including filename) to the image_selectivity_data.csv file. By default, this is just 'image_selectivity_dataframe.csv', which assumes that the file is in the same directory as the script you are running.

areas : We will restrict our attention to the areas in this list. By default this is the list ['VISp'], because we are visual cortex chauvanists here at the Allen Institute.

discard_nana : One of the outputs of this function is a dataframe that's been filtered to discard rows that have NaN values in the columns specified here.

Output:
	boc : the BrainObservatoryCache object from AllenSDK, initialized using the data from the external harddrive.

	specimens_with_selectivity_S : A dataframe the includes the output of boc.get_cell_specimens as well as a column for the selectivity index S
		See Decoding Visual Inputs From Multiple Neurons in the Human Temporal Lobe, J. Neurophys 2007, by Quiroga et al.

	VISp_cells_with_numbers : a dataframe that filters specimens_with_selectivity_S by discarding rows with NaN values in the columns specified above.
"""
	# The dynamic brain database is on an external hard drive.
	drive_path = '/Volumes/Brain2016'
	if sys.platform.startswith('w'):
		drive_path = 'd:'
	manifest_path = os.path.join(drive_path, 'BrainObservatory/manifest.json')
	boc = BrainObservatoryCache(manifest_file=manifest_path)
	expt_containers = boc.get_experiment_containers()

	all_expts_df = pd.DataFrame(boc.get_ophys_experiments(stimuli=list(stimuli_to_use)))
	# this has headers:
	# age_days	cre_line	experiment_container_id	id	imaging_depth	session_type	targeted_structure

	specimens_df = pd.DataFrame(
		boc.get_cell_specimens(experiment_container_ids=all_expts_df.experiment_container_id.values))
	# this has headers:
	# area	cell_specimen_id	dsi_dg	experiment_container_id	imaging_depth	osi_dg	osi_sg	p_dg	p_ns	p_sg
	# pref_dir_dg	pref_image_ns	pref_ori_sg	pref_phase_sg	pref_sf_sg	pref_tf_dg	time_to_peak_ns	time_to_peak_sg
	# tld1_id	tld1_name	tld2_id	tld2_name	tlr1_id	tlr1_name

	# There's also a handy bit of data from Saskia, in the form of a measurement called S. See
	# Decoding Visual Inputs From Multiple Neurons in the Human Temporal Lobe, J. Neurophys 2007, by Quiroga et al


	selectivity_S_df = pd.read_csv(selectivity_csv_filename, index_col=0)
	selectivity_S_df = selectivity_S_df[['cell_specimen_id', 'selectivity_ns']]

	specimens_with_selectivity_S = specimens_df.merge(selectivity_S_df, how='outer', on='cell_specimen_id')

	# This is all cells in VISp that have a value for the specified parameters (i.e not NaN)
	VISp_cells_with_numbers = specimens_with_selectivity_S
	for area_name in areas:
		VISp_cells_with_numbers = VISp_cells_with_numbers[VISp_cells_with_numbers.area == area_name]
	for col_name in discard_nan:
		VISp_cells_with_numbers = VISp_cells_with_numbers[np.isnan(VISp_cells_with_numbers[col_name]) == False]

	return boc, specimens_with_selectivity_S, VISp_cells_with_numbers

def get_sweep_responses_ns(expt_ids,analysis_directory = "BrainObservatory/ophys_analysis/", session = "B"):
	"""Reads cell sweep response from the h5 files on the external harddrive
Input:
	expt_ids : list of experiments to get responses for.

Output:
	sweep_responses : a dictionary with experiment ids as keys and sweep responses as values
	mean_sweep_responses : same as above, but with mean sweep responses."""
	# The dynamic brain database is on an external hard drive.
	drive_path = '/Volumes/Brain2016'
	sweep_responses = {}
	mean_sweep_responses = {}
	if sys.platform.startswith('w'):
		drive_path = 'd:'
	for e_id in expt_ids:
		path = os.path.join(drive_path,analysis_directory,str(e_id) + '_three_session_' + session + '_analysis.h5')
		# f = h5py.File(path)
		sweep_responses[e_id] = pd.read_hdf(path, 'analysis/sweep_response_ns')
		mean_sweep_responses[e_id] = pd.read_hdf((path, 'analysis/mean_sweep_responses_ns'))
	return sweep_responses, mean_sweep_responses


# Things I will want eventually:
# 1. A dataframe with columns for...
#	a. all the identifiers from the brain observatory
#	b. the preferred directions from the grating and place field stimuli
# 2. A numpy array which is N_neurons x N_timesamples, where entry i,j is 1 or 0 based on whether
#	 the neuron was "firing" at that point (how we decide that we'll get to later

# Data frames first:
# expt_containers_df = pd.DataFrame(expt_containers)

# The stimuli I'm interested in are basically everything except the natural movie and natural scenes
# If we want other stimuli, add them here.
# non_movie_stimuli = ['drifting_gratings', 'locally_sparse_noise', 'spontaneous', 'static_gratings']

# all_expts_df = pd.DataFrame(boc.get_ophys_experiments(stimuli=non_movie_stimuli))
# this has headers:
# age_days	cre_line	experiment_container_id	id	imaging_depth	session_type	targeted_structure

# seems like I can use get_cell_speciments to get everything I'm after
# specimens_df = pd.DataFrame(boc.get_cell_specimens(experiment_container_ids=all_expts_df.experiment_container_id.values))
# this has headers:
# area	cell_specimen_id	dsi_dg	experiment_container_id	imaging_depth	osi_dg	osi_sg	p_dg	p_ns	p_sg
# pref_dir_dg	pref_image_ns	pref_ori_sg	pref_phase_sg	pref_sf_sg	pref_tf_dg	time_to_peak_ns	time_to_peak_sg
# tld1_id	tld1_name	tld2_id	tld2_name	tlr1_id	tlr1_name

# There's also a handy bit of data from Saskia, in the form of a measurement called S. See
# Decoding Visual Inputs From Multiple Neurons in the Human Temporal Lobe, J. Neurophys 2007, by Quiroga et al


# selectivity_S_df = pd.read_csv(os.path.join(drive_path, '/BrainObservatory/image_selectivity_dataframe.csv'), index_col=0)
# selectivity_S_df = selectivity_S_df[['cell_specimen_id', 'selectivity_ns']]

# specimens_with_selectivity_S = specimens_df.merge(selectivity_S_df, how='outer', on='cell_specimen_id')

# # This is all cells in VISp that have a value for the specified parameters (i.e not NaN)
# # Discards rows NaN in the columns specified below.
# discard_nan = [
# 	'selectivity_ns',
# 	'osi_sg',
# 	'osi_dg',
# 	'time_to_peak_ns',
# 	'time_to_peak_sg',
# ]
# VISp_cells_with_numbers = specimens_with_selectivity_S[specimens_with_selectivity_S.area == 'VISp']
# for col_name in discard_nan:
# 	VISp_cells_with_numbers = VISp_cells_with_numbers[np.isnan(VISp_cells_with_numbers[col_name]) == False]
