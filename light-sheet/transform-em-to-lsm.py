import skimage.io
import numpy as np
import pandas as pd
from tps import ThinPlateSpline
from skimage.measure import regionprops

# Define data parameters
lsm_spatial_resolution = np.array([1.7, .406, .406])

# Load EM soma
print("Loading EM soma segmentations...")
em_labels = skimage.io.imread("../data/em-lsm-data/em-soma-segmentation.tif")
print("Loaded.")

# Get label centroids
print("Getting EM soma locations...")
regions = regionprops(em_labels)
em_labels_centroids = np.array([r.centroid for r in regions])


# ------------------------------------------------------
# Transform EM points into LSM space
# ------------------------------------------------------

print("Fitting Thin Plate Spline transformation...")

# Read landmarks (with EM and LSM coordinates)
lsm2em_landmarks = pd.read_csv(f"../data/em-lsm-data/LSM2EMLandmarks.csv", header=None)
columns = ["name", "validity", "lsm_x", "lsm_y", "lsm_z", "em_x", "em_y", "em_z"]
lsm2em_landmarks.columns = columns
lsm2em_landmarks = lsm2em_landmarks[lsm2em_landmarks["validity"]==True] # Get valid landmarks only

# Get EM and LSM points
em_points = lsm2em_landmarks[["em_z", "em_y", "em_x"]].values
lsm_points = lsm2em_landmarks[["lsm_z", "lsm_y", "lsm_x"]].values

# Convert LSM points to voxels
lsm_points_voxels = lsm_points.copy() / (lsm_spatial_resolution*1000)

# Fit EM-to-LSM TPS transformation object
em2lsm = ThinPlateSpline(alpha=0.0)  # 0 Regularization
em2lsm.fit(em_points, lsm_points_voxels)

# Transform all EM points
em_labels_centroids_lsm_trans = em2lsm.transform(em_labels_centroids)