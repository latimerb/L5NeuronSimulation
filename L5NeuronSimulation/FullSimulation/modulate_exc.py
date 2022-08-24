import h5py
import pandas as pd
from raster_maker import SonataWriter
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy

df = pd.read_csv('Connections.csv')

p_delta = 0.1 # change p_delta of the nodes to 100% modulation
num_cells = int(p_delta * df[(df['Source Population']=='exc_stim')&(df.Name.str.contains('dend'))]['Node ID'].nunique())
cells_to_change = np.random.choice(df[(df['Source Population']=='exc_stim')&
                                      (df.Name.str.contains('dend'))]['Node ID'].unique(),
                                   num_cells,
                                   replace=False)

depth_of_mod = 1
freq = 64
phase= 0#2*np.pi/3
tsim = 400 # seconds
t = np.arange(0,tsim,.001)

mod_trace = depth_of_mod*(np.sin((2 * np.pi * freq * t ) - phase) + 1) + (1-depth_of_mod)

#numbPoints = scipy.stats.poisson(rate_temp/1000).rvs()#Poisson number of points
#simSpks=np.where(numbPoints>0)[0]

f = h5py.File('exc_stim_spikes.h5','r')

mask = np.isin(f['spikes']['exc_stim']['node_ids'][:], cells_to_change)
anti_mask = ~mask

old_timestamps = f['spikes']['exc_stim']['timestamps'][anti_mask]
old_nodeids = f['spikes']['exc_stim']['node_ids'][anti_mask]

fr_df = pd.DataFrame(np.concatenate((f['spikes']['exc_stim']['timestamps'][mask].reshape(-1,1),
                                     f['spikes']['exc_stim']['node_ids'][mask].reshape(-1,1)),axis=1),
             columns = ['timestamps','node_ids'])
fr_df = (fr_df.groupby('node_ids')['timestamps'].count()/tsim).reset_index()

ts = []
nid = []
for n in fr_df['node_ids']:
    fr = fr_df.loc[fr_df.node_ids==n,'timestamps'].values
    #import pdb; pdb.set_trace()
    numbPoints = scipy.stats.poisson(fr*mod_trace/1000).rvs()
    ts.append(np.where(numbPoints>0)[0])
    nid.append(np.repeat(n,np.where(numbPoints>0)[0].shape[0]))

new_timestamps = np.concatenate(ts).ravel()
new_nodeids = np.concatenate(nid).ravel()

timestamps = np.concatenate((old_timestamps,new_timestamps)).astype(int)
node_ids = np.concatenate((old_nodeids,new_nodeids)).astype(int)

fname = 'exc_stim_spikes2.h5'
writer = SonataWriter(fname, ["spikes", "exc_stim"], ["timestamps", "node_ids"], [np.float, np.int])

for i in np.unique(node_ids):
    simSpks = timestamps[node_ids==i]
    writer.append_repeat("node_ids", int(i), len(simSpks))
    writer.append_ds(simSpks, "timestamps")
    
writer.close()
