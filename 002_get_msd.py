import nd2
import trackpy as tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

file_name = 'data/micro1.nd2'
tp.quiet(suppress=True)  # Suppress output

# Parameters - adjust as needed
diameter = 3  # for feature finding
minmass = 1
search_range = 5  # max displacement between frames
memory = 3  # frames a particle can disappear
min_track_len = 5  # min length for trajectories
mpp = 0.1  # microns per pixel
fps = 10  # frames per second
n_lags_fit = 5  # number of initial lags to fit for D

# Read ND2
f = nd2.ND2File(file_name)
frames = f.asarray()[:100,:500,:500,:]
print("frames size",frames.shape)

# Assume it's a 3D array: time, y, x (grayscale single channel)
if frames.ndim == 4:
    if f.sizes.get('C', 1) > 1:
        frames = frames[:, 1, :, :]  # take first channel

# Get metadata if possible
try:
    voxel = f.voxel_size()
    mpp = voxel.x  # assume isotropic in xy
except:
    pass

# For fps, try to compute
try:
    events = f.events(orient='records')
    times = [e['Relative Time (s)'] for e in events]
    dt = np.mean(np.diff(times))
    fps = 1 / dt
except:
    pass
print("start tracking")
# Locate features
features = tp.batch(frames, diameter=diameter, minmass=minmass,processes=1)
print("tracked particles",len(features))
