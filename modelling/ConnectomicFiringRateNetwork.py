import dill
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------------------------
# Create Firing Rate network
# -------------------------------------------

def dillpickler(fname:str, mode="pickle", data=None):
    """Pickle or unpickle objects"""
    assert mode=="pickle" or mode=="unpickle", "Mode must be either 'pickle' or 'unpickle'"
    openmode = "wb" if mode=="pickle" else "rb"
    with open(fname, openmode) as f:
        if mode=="pickle":
            # assert len(data) != 0, "Data must be provided"
            obj = dill.dump(data, f)
        elif mode=="unpickle":
            obj = dill.load(f)
        return obj

# Function to normalise weights
def normW(weights, scale=1):
    weights_norm = weights.clone()
    N0 = weights_norm.shape[0]
    for ii in range(N0):
        if torch.sum(weights_norm[:,ii] > 0):
            weights_norm[:,ii] = weights_norm[:,ii] / torch.sqrt(torch.sum(weights_norm[:,ii]**2))
    return scale*weights_norm

def getTimeString():
    date = datetime.now()
    rand_str = str(np.random.rand())[-4:]
    return f"{date.day}_{date.month}_{date.year}_{date.hour}{date.minute}_{rand_str}"

class FiringRateNetwork(torch.nn.Module):
    """
    A firing rate network model for simulating neural activity.
    
    Attributes:
        dt (float): Time step size for simulation
        batch_size (int): Number of parallel simulations
        baseline (float): Baseline firing rate
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(
        self,
        dt: float = 0.5,
        batch_size: int = 1,
        baseline: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.dt = dt
        self.batch_size = batch_size
        self.baseline = baseline
        self.device = device
        self.neuron_names = []
        self.recordings = {}
        self.time_string = getTimeString()

    # -----------------------------------------------
    # Setup
    # -----------------------------------------------
    def loadWeightsFromEM(self):
        self.weights = pd.read_csv(f"../data/connectome-data/ad_2022-03-17.csv", index_col=0)
    
    def initialiseNetwork(self):
        # Initialise parameters for all neurons
        self.potentials = 0 * torch.ones(self.n_neurons, self.batch_size, dtype=torch.float32)
        # self.adaptation = 0 * torch.ones(self.n_neurons, self.batch_size, dtype=torch.float32)
        self.I_ext = 0 * torch.ones(self.n_neurons, self.batch_size, dtype=torch.float32)
        self.effective_weights = torch.zeros((self.n_neurons, self.n_neurons), dtype=torch.float32, device=self.device)
    
    def initialiseParameters(self, neuron_sign_sigma=.01):
        # Initialise parameters for all paired neurons (i.e. with one value for each pair)
        '''NOTE: Number of paired neurons is taken from the left hemisphere'''
        self.initialiseNetwork()
        self.weight_scalar = torch.ones([1], dtype=torch.float32, device=self.device)
        self.baselines = self.baseline*torch.ones(self.n_neurons,1, dtype=torch.float32, device=self.device)
        self.timescales = 1*torch.ones(self.n_neurons,1, dtype=torch.float32, device=self.device)

    def getPrePostWeights(self, pres, posts):
        return self.weights[pres,:][:,posts]
    
    def subsetNetwork(self, skids_of_interest):
        # Filter SKIDs to only those already in network
        network_weight_skids = self.weights.index.astype(int).tolist()
        skids_of_interest = [skid for skid in skids_of_interest if skid in network_weight_skids]
        # Get subset of weights
        self.weights = self.weights.loc[skids_of_interest,:][[str(s) for s in skids_of_interest]]
        self.neuron_skids = self.weights.index.astype(int).tolist()
        # Convert Pandas weights to Torch tensors
        self.weights = torch.from_numpy(self.weights.T.values.astype(np.float32))
        self.n_neurons = len(skids_of_interest)
    
    def setInhibitoryNeurons(self, inhibitory_neuron_skids, inhibitory_neuron_indices):
        self.inhibitory_neuron_skids = inhibitory_neuron_skids
        self.inhibitory_neuron_indices = inhibitory_neuron_indices
    
    # -----------------------------------------------
    # Dynamics
    # -----------------------------------------------
    nonlinear = lambda self, x : torch.tanh(x)
    sigmoid = lambda self, x, a, b : (.5 + (torch.tanh(a*(x-b))/2))
    def updatePotentials(self):
        self.I = torch.mm(self.effective_weights, self.potentials) + self.baselines + self.I_ext
        # self.adaptation = self.adaptation + self.dt * (\
        #     -(self.adaptation / self.tau_adapt) +
        #     (self.potentials / self.tau_adapt)
        self.potentials = self.potentials + self.dt * (\
            - (self.potentials / self.timescales) \
            + self.nonlinear(self.I) \
            # - (self.adaptation_weight * self.adaptation)
        )
    
    def initialiseRecordings(self):
        # Initialise recordings
        self.recordings = {}
        self.recordings['potentials'] = torch.zeros(size=[self.T, self.n_neurons, self.batch_size])
        self.recordings["I"] = torch.zeros(size=[self.T, self.n_neurons, self.batch_size])
        self.recordings["I_ext"] = torch.zeros(size=[self.T, self.n_neurons, self.batch_size])

    def updateRecordings(self, t):
        self.recordings['potentials'][t,:,:] = self.potentials
        self.recordings['I'][t,:,:] = self.I
        self.recordings['I_ext'][t,:,:] = self.I_ext

    # -------------------------------------------
    # Running functions
    # -------------------------------------------

    def getEffectiveWeights(self):
        self.effective_weights = (self.weights * self.neuron_signs) * self.weight_scalar

    def setStimulationTimecourse(self, I_exts):
        self.T = I_exts.shape[1]
        self.I_exts = I_exts
    
    def runSimulation(self):
        for t in range(self.T):
            self.I_ext = self.I_exts[:,t]
            self.updatePotentials()
            self.updateRecordings(t)

    # -------------------------------------------
    # Training functions
    # -------------------------------------------

    def setTrainingParameters(self):
        self.training_parameters = [
            self.weight_scalar,
            self.neuron_sign_inputs,
            self.baselines,
            self.timescales,
            # self.adaptation_weight,
            # self.tau_adapt,
        ]
        for i in range(len(self.training_parameters)):
            self.training_parameters[i].requires_grad = True

    def enforceConstraints(self):
        with torch.no_grad(): # so Torch does not calculate gradients with respect to these operations
            self.neuron_sign_inputs[self.inhibitory_neuron_indices] = -15 # Inhibitory neurons should have negative signs
            # Baseline should always be greater than 0
            self.baselines = torch.clamp(self.baselines, min=0)
    
    # -----------------------------------------
    # Set device
    # -----------------------------------------

    def setDevice(self):
        # Set training parameters
        if self.device == "cuda":
            self.weight_scalar = self.weight_scalar.cuda()
            self.neuron_sign_inputs.data = self.neuron_sign_inputs.data.cuda()
            self.timescales.data = self.timescales.data.cuda()
            self.baselines.data = self.baselines.data.cuda()
        elif self.device == "cpu":
            self.neuron_sign_inputs.data = self.neuron_sign_inputs.data.cpu()
            self.timescales.data = self.timescales.data.cpu()
            self.baselines.data = self.baselines.data.cpu()
        # Set other variables
        self.weights = self.weights.to(self.device)
        self.I_exts = self.I_exts.to(self.device)
        self.potentials = self.potentials.to(self.device)
        self.effective_weights = self.effective_weights.to(self.device)
        for key in self.recordings.keys():
            self.recordings[key] = self.recordings[key].to(self.device)
    
    # -----------------------------------------
    # Saving
    # -----------------------------------------
    def loadPickled(self, save_path):
        '''Read from pickled Experiment object'''
        try:
            # print("Loading experiment from pickled...")
            exp = dillpickler(f"{save_path}", "unpickle")
            self.__dict__.update(exp.__dict__)
        except:
            raise RuntimeError("Pickled Experiment object could not be found")

    def pickleNetwork(self, save_path):
        dillpickler(f"{save_path}.net", "pickle", self)


