import matplotlib.pyplot as plt
import numpy as np
# from skimage import exposure
# from skimage.feature import blob_log
import nd2,trackpy as tp

filename = "mainfolder8/DF_Au50_5%biot_2928x512_pos02.nd2"

# --- LOCATE PARAMS (you choose) ---
diameter       = 5          # odd integer ~ feature size in pixels
minmass        = 60         # brightness filter; raise to remove noise
separation     = 5          # min distance between features (>= diameter to prevent doubles)
threshold      = 2           # baseline intensity threshold
invert         = False       # True if particles are dark on bright background
noise_size     = 1          # pixel smoothing (Gaussian in preprocess)
smoothing_size = 0           # set 0 or None to disable smoothing
percentile     = 50          # robust background subtraction; 0 disables
# --- LINKING PARAMS ---
search_range = 9             # max displacement (px) between frames
memory       = 3             # allow missing detections up to N frames


#%%time
subarr=None
with nd2.ND2File(filename) as f:
    print("Sizes:", f.sizes)           # e.g. {'T':10,'C':2,'Z':5,'Y':2048,'X':2048}
    print("Channels:", [str(ch) for ch in f.metadata.channels])
    subarr = f.asarray()                  # numpy array in native order (often TCZYX)
#%%time
# normalize by dividing by 1000
#subarr = arr.copy()#[:,:500,1000:1500]
frames = (np.clip(subarr/subarr.max(),0,1)*255).astype("uint8")
del subarr
#%%
def weighted_gray(x: np.ndarray, channel_axis: int = -1, out_dtype=None):
    """Return grayscale stack from RGB using 0.299R + 0.587G + 0.114B.
    x: 4D array, e.g. (T,H,W,3) or (T,3,H,W)
    channel_axis: -1 for ...x3, 1 for 3x..."""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    rgb = np.moveaxis(x, channel_axis, -1)[..., :3].astype(np.float32, copy=False)
    gray = rgb @ w  # shape: x without the channel axis

    if out_dtype is not None:
        if np.issubdtype(out_dtype, np.integer):
            # clip to the dtype’s valid range if converting to ints
            lo, hi = (0, np.iinfo(out_dtype).max)
            gray = np.clip(gray, lo, hi)
        gray = gray.astype(out_dtype)
    return gray
#%%
gray1 = weighted_gray(frames, channel_axis=-1, out_dtype=np.uint8)

#%%time
# --- DETECT (batch over stack) ---
loc = tp.batch(
    gray1,
    diameter=diameter,
    minmass=minmass,
    separation=separation,
    threshold=threshold,
    invert=invert,
    preprocess=True,
    noise_size=noise_size,
    smoothing_size=(smoothing_size or None),
    percentile=percentile,
)
#%%time
# --- LINK INTO TRAJECTORIES ---
traj = tp.link_df(loc, search_range=search_range, memory=memory)

# Optional: drop very short tracks
traj = tp.filter_stubs(traj, threshold=5)  # keep tracks with ≥5 points
traj.index = traj.index.rename('frame_idx')
df = traj.sort_values(['particle','frame']).copy()
groups = df.groupby('particle', sort=True)
#%%time
fig,axx=plt.subplots(2,1,figsize=(15,6),dpi=200)
axx[0].imshow(np.clip(np.stack([gray1[0],gray1[-1],gray1[-1]*0],axis=-1)/20-1,0,1),interpolation="none",vmax=20)
axx[0].set_title("Tracked particles (Red: before, Green: after)")
axx[0].set_xlabel("X position (pixels)")
axx[0].set_ylabel("Y position (pixels)")
ax=axx[1]
ax.imshow(gray1[0],origin="upper",interpolation="none",vmax=50)
for i, (pid, g) in enumerate(groups):
    ax.plot(g['x'].values, g['y'].values, '-', lw=1,c="w")
ax.set_title("Particle trajectories over first frame")
ax.set_xlabel("X position (pixels)")
ax.set_ylabel("Y position (pixels)")

# Save and close
plt.tight_layout()
plt.savefig("particle_tracking_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)