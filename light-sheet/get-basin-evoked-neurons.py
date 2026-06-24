import os
import numpy as np
import skimage.io
import pandas as pd
from tqdm import tqdm
import matplotlib.pylab as plt
from skimage.measure import regionprops
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp

# Load data
all_neuron_traces = np.load("./data/neuron_traces_cleaned_3um.npz")["neuron_traces"]
brain_neuron_indices = np.load("./data/indices_of_brain_neurons.npy")
behaviour_annotations = pd.read_csv(f"./data/eventAnnotations.csv")
segmentation = skimage.io.imread(f"./data/em_based_lsm_segmentation_3um.tif")
neuron_traces_em_locations = pd.read_csv("./data/neuron_traces_em_locations.csv", header=None).values

# Get segmentation centroids
regions = regionprops(segmentation)
em_based_lsm_labels_idx = np.array([r.label for r in regions]) - 1 # get neuron labels
em_based_lsm_labels_centroids = np.array([r.centroid for r in regions]) # get neuron centroids
em_based_lsm_labels_areas = np.array([r.area for r in regions]) # get neuron areas


# ------------------------------------------------------
# Get times of behaviour and stimulation events
# ------------------------------------------------------

# Get behaviour times
behaviourData = behaviour_annotations.to_numpy()
eventNames = ["forward", "backward", "stim", "hunch", "other", "left_turn", "right_turn", "HP"]
eventTimes = {e: [] for e in eventNames}
for b in behaviourData:
    start, end, forward, backward, stim, hunch, other, turn, left_turn, right_turn, HP = [int(c) if len(c)>0 else 0 for c in b[0].split(";")]
    cur_events = [forward, backward, stim, hunch, other, left_turn, right_turn, HP]
    assert np.sum(cur_events) == 1, "Multiple events in one timepoint"
    event_name = eventNames[cur_events.index(1)]
    # Subtract 1 from times (to use 0 indexing)
    start, end = start-1, end-1
    eventTimes[event_name].append([start, end])

# ------------------------------------------------------------
# Get evoked data
# ------------------------------------------------------------

# Limit analyses to brain neurons
neuron_traces = all_neuron_traces[brain_neuron_indices]

# Set windows
preWindow = 15 # average for F0
onWindow = 10
after_event_Window = 15 # frames after event stop that are considered
n_neurons, n_timepoints = neuron_traces.shape

# Get evoked data
cur_events = eventTimes["stim"]
eventDurations = np.array([np.ptp(e) for e in cur_events])
postWindow = np.max(eventDurations) + after_event_Window

# Get only those events that start after preWindow, and end before n_timepoint - postWindow
cur_events = [e for e in cur_events if e[0] > preWindow and e[1] < n_timepoints-postWindow]

# Calculate DF / F0 for each instance of event 
evokedData = np.zeros([len(cur_events), n_neurons, preWindow+postWindow])
for i,e in enumerate(tqdm(cur_events)):
    start, end = e
    cur_evoked = neuron_traces[:,start-preWindow:start+postWindow].copy()
    baseline_F0 = np.nanmean(cur_evoked[:,:preWindow],axis=1)[:,None] # make baseline habe shape of (n_neurons, 1)
    # Calculate delta F over F0
    cur_evoked_deltaFOverF = (cur_evoked - baseline_F0) / baseline_F0
    evokedData[i] = cur_evoked_deltaFOverF

# ------------------------------------------------------------
# Find hits
# ------------------------------------------------------------

# Get on responses
def getOnResponses(data, preWindow, onWindow):
    return np.nanmean(data[:, :, preWindow:preWindow+onWindow],axis=2)

# Get on period responses
onResponses = getOnResponses(evokedData, preWindow, onWindow)
meanOnResponses = onResponses.mean(axis=0)

# Get p-values
p_values = []
for i in range(onResponses.shape[1]):
    t, p = ttest_1samp(onResponses[:,i], 0)
    p_values.append(p)

# Get corrected p-value thresholds
method = "fdr_by"
corrected_hits, pvals_corrected, _, _ = multipletests(p_values, alpha=.1, method=method)
hit_neurons_sig = np.where(corrected_hits==True)[0]
print(f"Number of significance hits: {np.sum(corrected_hits)}")

# Get standard devaition hits
n_stds = 3
std_threshold = meanOnResponses.mean() + (n_stds * meanOnResponses.std(axis=0))
hit_neurons_std = np.where(meanOnResponses > std_threshold)[0]
print(f"Number of {n_stds} STD hits: {len(hit_neurons_std)}")

# Define hit neurons
hit_neurons = hit_neurons_sig

# Remove neurons with small areas
areaThreshold = em_based_lsm_labels_areas.mean() - em_based_lsm_labels_areas.std()
hit_neurons_areas = em_based_lsm_labels_areas[brain_neuron_indices][hit_neurons]
hit_neurons = hit_neurons[hit_neurons_areas > areaThreshold]
print(f"Number of hit neurons = {len(hit_neurons)}")

# Get hit locations and mean timecourses
hit_timecourses = evokedData[:,hit_neurons].mean(axis=0)
hit_locations_em = neuron_traces_em_locations[hit_neurons]

# ------------------------------------------------------------
# Plot hit neurons
# ------------------------------------------------------------

# Make figures folder
os.makedirs("./figures", exist_ok=True)

# Plot results
plt.close("all")
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
title = f"Number of responding cell = {len(hit_neurons)}\nThreshold = {n_stds} STDs"
all_soma_centroids_to_plot = neuron_traces_em_locations
Xs = np.linspace(-preWindow, postWindow, hit_timecourses.shape[1])
ax[0].scatter(-all_soma_centroids_to_plot[:,1], all_soma_centroids_to_plot[:,0], color="black", alpha=.15, s=5)
ax[1].scatter(all_soma_centroids_to_plot[:,2], all_soma_centroids_to_plot[:,1], color="black", alpha=.15, s=5)
ax[2].scatter(all_soma_centroids_to_plot[:,2], all_soma_centroids_to_plot[:,0], color="black", alpha=.15, s=5)
for i in range(len(hit_locations_em)):
    ax[0].scatter(-hit_locations_em[i,1], hit_locations_em[i,0], s=50)
    ax[1].scatter(hit_locations_em[i,2], hit_locations_em[i,1], s=50)
    ax[2].scatter(hit_locations_em[i,2], hit_locations_em[i,0], s=50)

ax[-1].plot(Xs, hit_timecourses.T, alpha=1)
yLims = ax[-1].get_ylim()
ax[-1].fill_between([0, Xs[preWindow+onWindow]], yLims[0], yLims[1], color="black", alpha=.1)
plt.suptitle(title)
plt.tight_layout()
plt.savefig(f"./figures/hitResponders.png", dpi=500)

# ------------------------------------------------------------
# Plot all traces for hit neurons
# ------------------------------------------------------------

# Plot all neuron traces
plt.close("all")
n_rows = 8; n_cols = int(np.ceil(len(hit_neurons) / n_rows))
all_hit_timecourses = evokedData[:,hit_neurons]
fig, ax = plt.subplots(n_rows, n_cols, figsize=(10*n_rows, 5*n_cols))
cmap = plt.get_cmap("Greys")
colours = cmap(np.linspace(0,1,all_hit_timecourses.shape[0]))
ax = ax.flatten()
for i in tqdm(range(len(hit_neurons))):
    ax[i].plot(all_hit_timecourses[:,i].T, alpha=.5)
    ax[i].plot(all_hit_timecourses[:,i].mean(axis=0), linewidth=2, color="black")
    ax[i].set_title(f"Neuron {hit_neurons[i]}")

plt.tight_layout()
plt.savefig("./figures/all_hit_neuron_traces.png")