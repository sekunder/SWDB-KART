# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 16:11:22 2016

@author: ttruszko
"""
# In[]
# general initialization
from data_preprocessing import * 
boc, Selectivity_S_df, y = BOC_init(selectivity_csv_filename='C:\\Users\\ttruszko\\Documents\\GitHub\\SWDB-KART\\image_selectivity_dataframe.csv')    
import pandas as pd
import seaborn as sns
sns.set_context("notebook", font_scale=6.5,rc={"lines.linewidth": 6.5})

#from allensdk.core.brain_observatory_cache import BrainObservatoryCache
drive_path='d:'
manifest_path = os.path.join(drive_path,'BrainObservatory/manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)

# In[]
# BE CAREFUL- TAKES A zillion YEARS TO RUN
from allensdk.brain_observatory.natural_scenes import NaturalScenes

# In[]
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
# Get data

#expt_cont_ids=[511510733, 511511015]

targeted_structures = boc.get_all_targeted_structures()
#targeted_structure=targeted_structures[3]     
expt_cont_list = pd.DataFrame(boc.get_experiment_containers())

all_raw_reliability=[]
for struct in targeted_structures:
    expt_cont_VISlist=expt_cont_list[expt_cont_list['targeted_structure']==struct]
    expt_cont_ids=expt_cont_VISlist['id'].unique()
    #print(expt_cont_VISlist)
    #print(expt_cont_ids)
    
    expcont_raw_reliability=[]
    for cont_id in expt_cont_ids:
        session_id, session_data = get_session_id(cont_id, 'B', boc=boc)
        print(session_id)
        response, _, _, _ = open_h5_file_sessionID(session_id, 'd:', 'B')
        #data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_id)
        
        print('NS done')
        #responses=NS.get_response()
        raw_reliability=response[:,:,2] / 50

        #keep_reliability = raw_reliability > 0
        keep_reliability_5 = raw_reliability >= 0.1
        raw_reliability[~keep_reliability_5] = np.nan
        expcont_raw_reliability.append(raw_reliability)
        print('loop done')
        raw_reliability=[]
        keep_reliability_5=[]
    all_raw_reliability.append(expcont_raw_reliability)

# In[]
# do the plotting for mean over structures

# take mean of all_raw_reliability 
mean_reliability_struct=[4, ]
for i in range(len(all_raw_reliability)):
    for j in range(len(all_raw_reliability[i])):
        mean_reliability_struct[i,j].append(np.nanmean(all_raw_reliability[i][j], axis=1))

# In[]

area_mean_reliability = {}
mean_reliability = {}
area_sem_reliability = {}
for area in range(len(all_raw_reliability)):
    tmp=[]
    for expt in range(len(all_raw_reliability[area])):
        across_images_mean = np.nanmean(all_raw_reliability[area][expt], axis=1)
        tmp.append(across_images_mean)
        mean_reliability[targeted_structures[area]] = tmp
    area_mean_reliability[targeted_structures[area]] = np.nanmean(mean_reliability[targeted_structures[area]],axis=0)
    sem = np.std(mean_reliability[targeted_structures[area]],axis=0)/np.sqrt(len(mean_reliability[targeted_structures[area]]))    
    area_sem_reliability[targeted_structures[area]] = sem
# In[]

n_images = area_sem_reliability['VISl'].shape[0]
x_range = np.arange(0,n_images)
fig,ax=plt.subplots()
colors = ['r','g','b','purple']
for i,area in enumerate(targeted_structures):
#    handles = ax.plot(x_range,area_mean_reliability[area],label=area)
    ax.errorbar(x=x_range,y=area_mean_reliability[area],yerr=area_sem_reliability[area],label=area)
ax.legend(loc='best', ncol=4)
ax.set_xlabel('Image Number')
ax.set_ylabel('Mean Reliability Across Populations')
ax.set_title('Mean Reliability over visual structures')
# In[]
#do the plotting for 1 expt cont

fig, ax = plt.subplots(1, 1)
ax.plot(np.sort(np.nanmean(all_raw_reliability[0][0], axis=1)), 'o', markersize=20, color = 'steelblue')
ax.set_ylabel('Mean Reliability')
ax.set_xlabel('Images')
#ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_title('Sorted Mean Reliability for one experiment container')
#plt.plot(np.nanmean(raw_reliability, axis=1), '.')

#reacquire a raw set of data
expt_cont_ids=[511510733]
for cont_id in expt_cont_ids:
    session_id, session_data = get_session_id(cont_id, 'B', boc=boc)
    print(session_id)
    response, _, _, _ = open_h5_file_sessionID(session_id, 'd:', 'B')
        #data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_id)
        
    print('NS done')
        #responses=NS.get_response()
    raw_reliability=response[:,:,2] / 50
        
plt.hist(raw_reliability[~np.isnan(raw_reliability)], bins=100)
plt.xlabel('Reliability')
plt.ylabel('Counts')
plt.axvline(x=0.1)
plt.title('Reliability per cell per image for one experiment container')

# In[]
#max reliability which image?
#rands=[9, 3, np.nan, 5, 17, 0]
#np.nanargmax(rands)

idx_of_maxR=np.nanargmax(all_raw_reliability[0][0][0])

# In[]
# figure out how many presentations I ditched by using 0.1 cutoff
nan_count=np.arange(len(all_raw_reliability))
for i in range(len(all_raw_reliability)):
    nan_count[i]=np.arange(len(all_raw_reliability[i]))
    for j in range(len(all_raw_reliability[i])):
        for k in range(len(all_raw_reliability[i][j])):
            nan_count[i][j][k] = np.count_nonzero(np.isnan(all_raw_reliability[i][j][k]))

# In[]
nan_fraction = {}
for area in range(len(all_raw_reliability)):
    tmp=[]
    for expt in range(len(all_raw_reliability[area])):
        nans = np.where(np.isnan(all_raw_reliability[area][expt])==True)[0].shape[0]
        total = all_raw_reliability[area][expt].shape[0]*all_raw_reliability[area][expt].shape[1]
        tmp.append(nans/float(total))
    nan_fraction[area] = np.mean(tmp)
  
# In[]
#plot nan_fraction
#fig, ax = plt.plot(nan_fraction[area])
