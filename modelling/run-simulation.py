import os
import dill
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.lines import Line2D
from ConnectomicFiringRateNetwork import FiringRateNetwork, normW

# Seed random number generator
np.random.seed(10)
torch.manual_seed(10)

# -------------------------------------------
# Run simulation
# -------------------------------------------

# Load predicted GABAergic neuron IDs
inhibitory_neuron_skids = np.loadtxt(f"../data/connectome-data/inhibitory_neuron_skids.txt").astype(int)

# Define Basin neurons IDs
basin_skids = [3034133, 3282869, 3041612, 3074106, 10179501, 3091943, 3040481, 4049878]

# Load hit neurons
hit_neuron_spreadsheet = "../data/em-lsm-data/Table2-V3-add1099_HitNeuronData.xlsx"
hit_neuron_data = pd.read_excel(hit_neuron_spreadsheet)
hit_neuron_skids = hit_neuron_data["skid Ref. brain"].values
hit_neuron_names = hit_neuron_data["Publication name"].values
indices_to_keep =[]
for i,skid in enumerate(hit_neuron_skids):
    if type(skid) == int:
        indices_to_keep.append(i)

hit_neuron_skids = hit_neuron_skids[indices_to_keep]
hit_neuron_names = hit_neuron_names[indices_to_keep]

# Define functionally inhibited neuron IDs
functionally_inhibited_neuron_skids  = [17075832, 7439913, 17728723, 17728730, 8809171, 11291653, 11263687, 4966994, 13849557]

# Initialise network
nn = FiringRateNetwork(device="cpu")
nn.loadWeightsFromEM()
nn.n_neurons = nn.weights.shape[0]
nn.initialiseParameters()

neuron_skids = nn.weights.index.values
basin_skids = np.array(basin_skids)
for skid in basin_skids:
    if skid not in neuron_skids:
        raise RuntimeError(f"Basin neuron {skid} not found in network")

# Get Basin skid indices
basin_indices = [list(nn.weights.index).index(int(skid)) for skid in basin_skids]

# Get hit neuron indices
hit_neuron_indices = []
hit_neuron_sample_skids = []
new_hit_neuron_skids = []
for skid in hit_neuron_skids:
    if skid in nn.weights.index:
        hit_neuron_indices.append(list(nn.weights.index).index(int(skid)))
        hit_neuron_sample_skids.append(skid)
        new_hit_neuron_skids.append(skid)
    else:
        print(f"Hit neuron {skid} not found in network")
    
hit_neuron_skids = new_hit_neuron_skids

# Get inhibitory neuron indices
inhibitory_neuron_indices = []
for skid in inhibitory_neuron_skids:
    if skid in nn.weights.index:
        inhibitory_neuron_indices.append(list(nn.weights.index).index(int(skid)))
    else:
        print(f"Inhibitory neuron {skid} not found in network")

# Define stimulus
simulation_length = 100
stimulus_start = 40
stimulus_end = 60
I_exts = torch.zeros(
    (nn.n_neurons, simulation_length, 1),
    dtype=torch.float32,
    device=nn.device
)
I_exts[basin_indices, stimulus_start:stimulus_end, 0] = 10
# I_exts[:, stimulus_start:stimulus_end, 0] = 10
nn.setStimulationTimecourse(I_exts)

# Convert weights dataframe to torch tensor
nn.weights = torch.from_numpy(nn.weights.T.values.astype(np.float32))
nn.weights = normW(nn.weights, scale=.5)

# # Shuffle connectome
# random_order = np.random.permutation(nn.n_neurons)
# nn.weights = nn.weights[random_order,:][:,random_order]

# Set neuron signs
nn.neuron_signs = torch.ones(nn.n_neurons, 1, dtype=torch.float32, device=nn.device)
nn.neuron_signs[inhibitory_neuron_indices] = -1

# Initialise network
nn.initialiseNetwork()
nn.initialiseRecordings()
nn.getEffectiveWeights()

# Run simulation
nn.runSimulation()

# Get evoked responses
n_t_relax = 30
evoked_responses = nn.recordings["potentials"][:,:,0]
evoked_responses -= evoked_responses[stimulus_start-10:stimulus_start,:].mean(axis=0)
hit_evoked_responses = evoked_responses[n_t_relax:, hit_neuron_indices]
hit_evoked_on_responses = hit_evoked_responses[stimulus_start-n_t_relax:stimulus_end-n_t_relax,:].mean(axis=0)

# Plot results
plt.close("all")
n_neurons_to_plot = hit_evoked_responses.shape[1]
n_rows = 5
consisent_y_axis = False
n_cols = int(np.ceil(n_neurons_to_plot/n_rows))
fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=consisent_y_axis, figsize=(7.5*n_cols, 5*n_rows))
ax = ax.flatten()
for i in range(n_neurons_to_plot):
    cur_skid = hit_neuron_sample_skids[i]
    if cur_skid in functionally_inhibited_neuron_skids :
        color = "red"
    else:
        color = "blue"
    ax[i].plot(hit_evoked_responses[:,i], linewidth=5, color=color)
    ax[i].set_title(f"({hit_neuron_names[i]}) {hit_neuron_sample_skids[i]}")

plt.tight_layout()
plt.savefig("./figures/hit_neuron_modelled_responses.png")

# Check match to observed data
excited_neuron_skids = [skid for skid in hit_neuron_skids if skid not in functionally_inhibited_neuron_skids ]
inhibited_neuron_indices = [list(hit_neuron_skids).index(skid) for skid in functionally_inhibited_neuron_skids ]
excited_neuron_indices =  [list(hit_neuron_skids).index(skid) for skid in excited_neuron_skids]
inhibited_evoked_on_responses = hit_evoked_on_responses[inhibited_neuron_indices]
excited_evoked_on_responses = hit_evoked_on_responses[excited_neuron_indices]
inhibited_evoked_on_responses_binarised = inhibited_evoked_on_responses > 0
excited_evoked_on_responses_binarised = excited_evoked_on_responses > 0

# Get accuracy
n_correct_inhibited = len(np.where(inhibited_evoked_on_responses_binarised == 0)[0])
n_correct_excited = len(np.where(excited_evoked_on_responses_binarised > 0)[0])
original_accuracy = (n_correct_inhibited + n_correct_excited) / len(hit_neuron_skids)
print(f"Accuracy: {original_accuracy}")

# Run Mann-Whitney U test
mannwhitneyu_comparison = stats.mannwhitneyu(inhibited_evoked_on_responses, excited_evoked_on_responses)
print(mannwhitneyu_comparison)

# Get mean exctited vs. inhibited traces
inhibited_hit_evoked_responses = hit_evoked_responses[:,inhibited_neuron_indices].numpy()
excited_hit_evoked_responses = hit_evoked_responses[:,excited_neuron_indices].numpy()

# Get means and SEM
inhibited_hit_evoked_responses_mean = np.median(inhibited_hit_evoked_responses, axis=1)
inhibited_hit_evoked_responses_sem = np.std(inhibited_hit_evoked_responses, axis=1)/np.sqrt(len(inhibited_hit_evoked_responses))
excited_hit_evoked_responses_mean = np.median(excited_hit_evoked_responses, axis=1)
excited_hit_evoked_responses_sem = np.std(excited_hit_evoked_responses, axis=1)/np.sqrt(len(excited_hit_evoked_responses))

plt.close("all")
plt.plot(inhibited_hit_evoked_responses_mean, linewidth=2, color="red", label="inhibited")
plt.plot(excited_hit_evoked_responses_mean, linewidth=2, color="blue", label="excited")
# plt.fill_between(range(len(inhibited_hit_evoked_responses_mean)), 
#     inhibited_hit_evoked_responses_mean - inhibited_hit_evoked_responses_sem, 
#     inhibited_hit_evoked_responses_mean + inhibited_hit_evoked_responses_sem, alpha=0.2, color="red")
# plt.fill_between(range(len(excited_hit_evoked_responses_mean)), 
#     excited_hit_evoked_responses_mean - excited_hit_evoked_responses_sem, 
#     excited_hit_evoked_responses_mean + excited_hit_evoked_responses_sem, alpha=0.2, color="blue")
plt.legend()
plt.savefig("./figures/inhibited_vs_excited_hit_evoked_responses.png")

# Plot inhibited vs excited
plt.close("all")
plt.figure()
plt.scatter(np.ones(len(inhibited_evoked_on_responses)), inhibited_evoked_on_responses, label="inhibited")
plt.scatter(2*np.ones(len(excited_evoked_on_responses)), excited_evoked_on_responses, label="excited")
plt.xticks([1,2], ["Inhibited", "Excited"])
plt.xlim([.5, 2.5])
plt.legend()
plt.savefig("./figures/inhibited_vs_excited_evoked_on_responses.png")


# --------------------------------------------------
# Run simulation with random weight shufflings
# --------------------------------------------------

# Initialis
os.makedirs("./data", exist_ok=True)
os.makedirs("./figures", exist_ok=True)

accuracy_store_fname = "./data/accuracy_store.npy"
if not(os.path.exists(accuracy_store_fname)):

    # Number of shufflings
    n_shufflings = 1000

    # Get original weights
    original_weights = nn.weights.clone()

    # Initialise statistics store
    statistics_store = []
    accuracy_store = []

    original_shape = nn.weights.shape
    original_weights_flat = original_weights.flatten()

    for i in tqdm(range(n_shufflings)):

        # # Shuffle connectome
        # nn.weights = original_weights_flat[np.random.permutation(len(original_weights_flat))]
        # nn.weights = nn.weights.reshape(original_shape)

        # Degree- or strength-preserving shuffle, connectome (shuffle within rows to preserve outgoing weights)
        nn.weights = original_weights.clone()
        for i in range(nn.weights.shape[0]):
            nn.weights[i, :] = nn.weights[i, torch.randperm(nn.weights.shape[1])]

        # Re-normalize
        nn.weights = normW(nn.weights, scale=.5)
            
        # Initialise network
        nn.initialiseNetwork()
        nn.initialiseRecordings()
        nn.getEffectiveWeights()
        
        # Run simulation
        nn.runSimulation()
        
        # Get evoked responses
        n_t_relax = 30
        evoked_responses = nn.recordings["potentials"][:,:,0]
        evoked_responses -= evoked_responses[stimulus_start-10:stimulus_start,:].mean(axis=0)
        hit_evoked_responses = evoked_responses[n_t_relax:, hit_neuron_indices]
        hit_evoked_on_responses = hit_evoked_responses[stimulus_start-n_t_relax:stimulus_end-n_t_relax,:].mean(axis=0)
        
        # Check match to observed data
        excited_neuron_skids = [skid for skid in hit_neuron_skids if skid not in functionally_inhibited_neuron_skids ]
        inhibited_neuron_indices = [list(hit_neuron_skids).index(skid) for skid in functionally_inhibited_neuron_skids ]
        excited_neuron_indices =  [list(hit_neuron_skids).index(skid) for skid in excited_neuron_skids]
        inhibited_evoked_on_responses = hit_evoked_on_responses[inhibited_neuron_indices]
        excited_evoked_on_responses = hit_evoked_on_responses[excited_neuron_indices]
        
        # Get stats
        r = stats.mannwhitneyu(inhibited_evoked_on_responses, excited_evoked_on_responses)
        statistics_store.append([r.statistic, r.pvalue])

        # Check match to observed data
        inhibited_evoked_on_responses_binarised = inhibited_evoked_on_responses > 0
        excited_evoked_on_responses_binarised = excited_evoked_on_responses > 0

        # Get accuracy
        n_correct_inhibited = len(np.where(inhibited_evoked_on_responses_binarised == 0)[0])
        n_correct_excited = len(np.where(excited_evoked_on_responses_binarised > 0)[0])
        accuracy = (n_correct_inhibited + n_correct_excited) / len(hit_neuron_skids)
        accuracy_store.append(accuracy)
    
    # Save
    np.save(accuracy_store_fname, accuracy_store)

else:

    # Load data
    accuracy_store = np.load(accuracy_store_fname)

# --------------------------------------------------
# Plot random accuracy values
# --------------------------------------------------

# Assess statistical significance of accuracy
accuracy_array = np.array(accuracy_store)
p_value = (1 + np.sum(accuracy_array >= original_accuracy)) / (1 + len(accuracy_array))
print(f"Permutation p-value: {p_value}")

# Plot results
plt.close("all")
fig = plt.figure()
plt.hist(100*np.array(accuracy_store), bins=50, width=2)
yLims = plt.ylim()
plt.vlines(100*original_accuracy, 0, yLims[1], linewidth=3, color="black", linestyle="-")
plt.ylim(yLims)
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.title(f"Distribution of accuracy (p = {p_value:.3f})")
plt.tight_layout()
plt.savefig("./figures/accuracy_distribution.png", dpi=300)



# --------------------------------------------------
# Plot statistics
# --------------------------------------------------

# # Get statistics
# statistics_store = np.array(statistics_store)
# statistics_store_stat = statistics_store[:,0]
# statistics_store_pvalue = statistics_store[:,1]

# # Plot distribution of statistics
# plt.close("all")
# plt.figure()
# plt.hist(statistics_store_stat, bins=50)
# plt.vlines(mannwhitneyu_comparison.statistic, 0, plt.ylim()[1], color="red", linestyle="--")
# plt.xlabel("Mann-Whitney U statistic")
# plt.ylabel("Frequency")
# plt.title("Distribution of Mann-Whitney U statistic")
# plt.savefig("./figures/mannwhitneyu_distribution.png")

# --------------------------------------------------
# Generate Manuscript Reporting String
# --------------------------------------------------

# 1. Format the Mann-Whitney U test results
u_stat = mannwhitneyu_comparison.statistic
mw_p = mannwhitneyu_comparison.pvalue

# 2. Format the Accuracy and Permutation P-Value
acc_percentage = original_accuracy * 100
perm_p = p_value

# 3. Print the formatted text for the manuscript
print("\n" + "="*50)
print("MANUSCRIPT STATISTICS REPORT")
print("="*50)
print(f"Mann-Whitney U Test:  U = {u_stat:.1f}, p = {mw_p:.4e}")
print(f"Network Accuracy:     {acc_percentage:.1f}%")
print(f"Permutation P-Value:  p = {perm_p:.4f}")
print("="*50 + "\n")

# --------------------------------------------------
# Generate 3-Panel Final Summary Figure
# --------------------------------------------------

# Set global aesthetic parameters for publication
plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2
})

# Create a 1x3 composite figure
plt.close()
fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1, 1.5]})

# Define colors
color_inhibited = '#D62728' # Brick Red
color_excited = '#1F77B4'   # Steel Blue

# ==========================================
# Panel A: All Simulated Traces (Z-scored)
# ==========================================
ax_A = axes[0]

# Convert traces to numpy array if they are tensors
traces = hit_evoked_responses.numpy() if hasattr(hit_evoked_responses, 'numpy') else hit_evoked_responses

# Z-score: mean zero in the first 5 timepoints
baseline_mean = np.mean(traces[:5, :], axis=0)
# Use standard deviation of the whole trace (adding small epsilon to prevent div by 0 for silent neurons)
trace_std = np.std(traces, axis=0) + 1e-8
z_scored_traces = (traces - baseline_mean) / trace_std

time_vector = np.arange(z_scored_traces.shape[0])

# Plot each trace individually
for i in range(z_scored_traces.shape[1]):
    # Determine color based on true in vivo label
    if i in inhibited_neuron_indices:
        color = color_inhibited
    else:
        color = color_excited
        
    ax_A.plot(time_vector, z_scored_traces[:, i], color=color, alpha=0.75, linewidth=1.5)

# Highlight stimulation window
stim_start_plot = stimulus_start - n_t_relax + 1
stim_end_plot = stimulus_end - n_t_relax + 1
ax_A.axvspan(stim_start_plot, stim_end_plot, color='grey', alpha=0.15, label='Basin Stimulation')

# Add custom legend for Panel A
custom_lines = [Line2D([0], [0], color=color_inhibited, lw=2, alpha=0.8),
                Line2D([0], [0], color=color_excited, lw=2, alpha=0.8)]
ax_A.legend(custom_lines, ['True Inhibited', 'True Excited'], loc='lower right', frameon=False, fontsize=10)

ax_A.set_xlabel('Simulation Timesteps')
ax_A.set_ylabel('Simulated Response (Z-score)')
ax_A.set_title('In Silico Temporal Dynamics', pad=15)

# ==========================================
# Panel B: ROC Curve (Ranking Ability)
# ==========================================
ax_B = axes[1]

# Combine data for ROC
inh_vals = np.array(inhibited_evoked_on_responses)
exc_vals = np.array(excited_evoked_on_responses)
all_vals = np.concatenate([inh_vals, exc_vals])

# True labels: 0 for inhibited, 1 for excited
true_labels = np.concatenate([np.zeros(len(inh_vals)), np.ones(len(exc_vals))])

# Calculate False Positive Rate, True Positive Rate, and thresholds
fpr, tpr, thresholds = roc_curve(true_labels, all_vals)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
ax_B.plot(fpr, tpr, color='#000000', lw=3, label=f'Model Performance\n(AUC = {roc_auc:.2f})')
ax_B.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance (AUC = 0.50)')

ax_B.set_xlim([-0.05, 1.05])
ax_B.set_ylim([-0.05, 1.05])
ax_B.set_xlabel('False Positive Rate')
ax_B.set_ylabel('True Positive Rate')
ax_B.set_title('In Silico Valence Ranking', pad=15)
ax_B.legend(loc="lower right", frameon=False, fontsize=10)

# Add MW-U p-value as text
ax_B.text(0.05, 0.95, f'Mann-Whitney U p = {mannwhitneyu_comparison.pvalue:.3f}', 
          transform=ax_B.transAxes, va='top', ha='left', fontsize=11)

# ==========================================
# Panel C: Permutation Testing (Connectome Specificity)
# ==========================================
ax_C = axes[2]

acc_array_pct = np.array(accuracy_store) * 100
true_acc_pct = original_accuracy * 100

counts, bins, patches = ax_C.hist(acc_array_pct, bins=30, color='lightgray', edgecolor='darkgray')
ax_C.axvline(true_acc_pct, color='black', linestyle='-', linewidth=2.5, 
             label=f'Biological Connectome\n({true_acc_pct:.1f}%)')

ax_C.set_xlabel('Strict Classification Accuracy (%)')
ax_C.set_ylabel('Frequency (Shuffled Networks)')
ax_C.set_title('Dependence on Wiring Topology', pad=15)

# ax_C.text(true_acc_pct - 1, np.max(counts) * 0.225, f'Permutation p = {p_value:.3f}', 
#           ha='right', va='top', fontsize=11)
ax_C.legend(frameon=False, loc='upper left', fontsize=10)

# Add panel labels (A, B, C)
for i, ax in enumerate(axes):
    ax.text(-0.15, 1.1, chr(65+i), transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='right')

# Save as high-resolution vector graphic and PNG
plt.savefig("./figures/Figure_Modelling_Final_Summary.pdf", dpi=300, bbox_inches='tight', format='pdf')
plt.savefig("./figures/Figure_Modelling_Final_Summary.png", dpi=300, bbox_inches='tight')
plt.show()