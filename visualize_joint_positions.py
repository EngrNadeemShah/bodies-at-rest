### 0. Initialization

# Import libraries
import sys
import torch
import smplx
import trimesh
import pyrender
import numpy as np
import random
import pickle
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import HDF5Dataset
from utils import visualize_smpl_with_joints

np.set_printoptions(threshold=sys.maxsize, precision=2, suppress=True)
# %matplotlib inline	# Uncomment this line if you are using Jupyter Notebook

# Adjust z-coordinate of the root joint (pelvis) by this amount to correct for elevation offset from FleX simulation settling
z_adj_scalar_meter	= -0.075		# -0.075m = -7.5cm = -75mm (-ve value means downwards shift)
z_adj_scalar_mm		= z_adj_scalar_meter * 1000

z_adj_1_meter = np.array([0, 0, z_adj_scalar_meter],	dtype=np.float32)
z_adj_1_mm    = np.array([0, 0, z_adj_scalar_mm],	dtype=np.float32)

z_adj_24_meter	= np.array(24 * [0, 0, z_adj_scalar_meter],dtype=np.float32).reshape(1, 24, 3)
z_adj_24_mm	= np.array(24 * [0, 0, z_adj_scalar_mm],	dtype=np.float32).reshape(1, 24, 3)



### 1. Load SMPL Model

# Path to SMPL models
smpl_feml_model_path_v1_0 = '/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'	# v1.0.0 has only 10 shape coefficients
smpl_male_model_path_v1_0 = '/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'	# v1.0.0 has only 10 shape coefficients
smpl_feml_model_path_v1_1 = '/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'	# v1.1.0 has 300 shape coefficients
smpl_male_model_path_v1_1 = '/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'	# v1.1.0 has 300 shape coefficients
smpl_neut_model_path_v1_1 = '/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'	# neutral is only available in v1.1.0

# Load SMPL models
model = smplx.SMPL(smpl_feml_model_path_v1_0)
# model = smplx.SMPL(smpl_male_model_path_v1_0)
# model = smplx.SMPL(smpl_feml_model_path_v1_1)
# model = smplx.SMPL(smpl_male_model_path_v1_1)
# model = smplx.SMPL(smpl_neut_model_path_v1_1)



### 2. Load .HDF5 (Pre-Processed) Dataset

# Path to hdf5 dataset
hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

# DataLoader setup
batch_size = 1
train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train', verbose=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Load an example from HDF5 dataset (Mod=1)
for inputs, true_labels in train_loader:
	# Extract SMPL parameters from true_labels
	joint_positions_hdf5_mm_adj = true_labels[:, 0:72].view(-1, 24, 3)	# Shape: (batch_size, 24, 3)
	betas_hdf5           		= true_labels[:, 72:82]     # Shape: (batch_size, 10)
	global_orient_hdf5   		= true_labels[:, 82:85]     # Shape: (batch_size, 3)
	body_pose_hdf5       		= true_labels[:, 85:154]    # Shape: (batch_size, 69)
	transl_hdf5_meter_adj		= true_labels[:, 154:157]   # Shape: (batch_size, 3)

	# Remove z_adj from root joint (pelvis) translation (position)
	transl_hdf5_meter			= transl_hdf5_meter_adj - z_adj_1_meter

	# Modify joint positions
	# Convert tensor to numpy array
	joint_positions_hdf5_mm_adj = joint_positions_hdf5_mm_adj.numpy()

	# Remove z_adj from joint positions
	joint_positions_hdf5_mm = joint_positions_hdf5_mm_adj - z_adj_24_mm

	# convert mm to m for fair comparison
	joint_positions_hdf5_meter = joint_positions_hdf5_mm / 1000.0

	# Remove z_adj from joint positions (meters)
	joint_positions_hdf5_meter_adj = joint_positions_hdf5_meter + z_adj_24_meter
	break



### 3. Load .p (Raw) Dataset

# 3.1. Pickle file paths

# crossed_legs (mod=1)
pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_f_lay_set2both_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_m_lay_set2both_4000.p'

# straight_limbs (mod=1)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000.p'

# # straight_limbs (mod=2)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'


# 3.2. Set a random index to visualize a specific example
# random_index = random.randint(0, num_examples - 1)
random_index = 0


# 3.3. Load the pickle file
with open(pkl_file_path, 'rb') as f:
	data = pickle.load(f, encoding='latin1')  # Use 'latin1' for Python 2 compatibility


# 3.4. Print the keys and shapes of the data loaded from the pickle file
# for i, (key, value) in enumerate(data.items(), start=1):
# 	print(f"({i:02}/{len(data):02}) {key}:\t{value.shape if isinstance(value, np.ndarray) else value.size() if isinstance(value, torch.Tensor) else np.array(value).shape if isinstance(value, list) else type(value)}")


# 3.5. Fetch the required data from the pickle file
joint_positions_pickled_meter	= np.array(data['markers_xyz_m'])[random_index].reshape(-1, 24, 3)	# Shape: (1, 24, 3)
betas_pickled					= torch.tensor(np.array(data['body_shape'])).float()[random_index].unsqueeze(0)				# Shape: (batch_size, 10)
global_orient_pickled			= torch.tensor(np.array(data['joint_angles'])).float()[:, 0:3][random_index].unsqueeze(0)	# Shape: (batch_size, 3)
body_pose_pickled				= torch.tensor(np.array(data['joint_angles'])).float()[:, 3:72][random_index].unsqueeze(0)	# Shape: (batch_size, 69)
transl_pickled_meter			= torch.tensor(np.array(data['root_xyz_shift'])).float()[random_index].unsqueeze(0)			# Shape: (batch_size, 3)


# 3.6.1. Add z_adj to root joint (pelvis) translation (position)
transl_pickled_meter_adj		= transl_pickled_meter + z_adj_1_meter

# 3.6.2. Adjust z to root joint position by the calculated values [0.0003, -0.2048, 0.0339]m
z_adj_1_meter_calc = np.array([0.0003, -0.2048, 0.0339], dtype=np.float32)  # Adjusted z for root joint (pelvis)
transl_pickled_meter_adj_calc = transl_pickled_meter - z_adj_1_meter_calc


# 3.7. Modify joint positions

# Add z_adj to joint positions (meter)
joint_positions_pickled_meter_adj = joint_positions_pickled_meter + z_adj_24_meter

# Convert from meters to millimeters
joint_positions_pickled_mm = joint_positions_pickled_meter * 1000

# Add z_adj to joint positions
joint_positions_pickled_mm_adj = joint_positions_pickled_mm + z_adj_24_mm



### 4. Compare __.HDF5__ and __.p__ Datasets

# 4.1. Print the shapes and values of the joint positions
print(f"___ Joint Positions Comparison __", end='\n\n')
print(f"joint_positions_hdf5_mm:	{joint_positions_hdf5_mm.shape}")
print(f"{joint_positions_hdf5_mm[0, 0:3, :]}", end='\n\n')
print(f"joint_positions_pickled_mm: {joint_positions_pickled_mm.shape}")
print(f"{joint_positions_pickled_mm[0, 0:3, :]}", end='\n\n')

print(f"joint_positions_hdf5_mm_adj:	{joint_positions_hdf5_mm_adj.shape}")
print(f"{joint_positions_hdf5_mm_adj[0, 0:3, :]}", end='\n\n')
print(f"joint_positions_pickled_mm_adj: {joint_positions_pickled_mm_adj.shape}")
print(f"{joint_positions_pickled_mm_adj[0, 0:3, :]}", end='\n\n')

print(f"joint_positions_hdf5_meter:	{joint_positions_hdf5_meter.shape}")
print(f"{joint_positions_hdf5_meter[0, 0:3, :]}", end='\n\n')
print(f"joint_positions_pickled_meter: {joint_positions_pickled_meter.shape}")
print(f"{joint_positions_pickled_meter[0, 0:3, :]}", end='\n\n')

print(f"joint_positions_hdf5_meter_adj:	{joint_positions_hdf5_meter_adj.shape}")
print(f"{joint_positions_hdf5_meter_adj[0, 0:3, :]}", end='\n\n')
print(f"joint_positions_pickled_meter_adj: {joint_positions_pickled_meter_adj.shape}")
print(f"{joint_positions_pickled_meter_adj[0, 0:3, :]}", end='\n\n')

# 4.2. Print the shapes of the SMPL parameters
print(f"___ SMPL Parameters Comparison __", end='\n\n')
print(f"betas_hdf5:		{betas_hdf5.shape}")
print(f"betas_pickled:		{betas_pickled.shape}", end='\n\n')

print(f"global_orient_hdf5:	{global_orient_hdf5.shape}")
print(f"global_orient_pickled:	{global_orient_pickled.shape}", end='\n\n')

print(f"body_pose_hdf5:		{body_pose_hdf5.shape}")
print(f"body_pose_pickled:	{body_pose_pickled.shape}", end='\n\n')

print(f"transl_hdf5:		{transl_hdf5_meter_adj.shape}")
print(f"transl_pickled:		{transl_pickled_meter.shape}", end='\n\n')

# 4.3. Print the values of the SMPL parameters
print(f"___ SMPL Parameters Comparison __", end='\n\n')
print(f"betas_hdf5:")
print(f"{betas_hdf5}")
print(f"betas_pickled:")
print(f"{betas_pickled}", end='\n\n')

print(f"global_orient_hdf5:	{global_orient_hdf5}")
print(f"global_orient_pickled:	{global_orient_pickled}", end='\n\n')

print(f"body_pose_hdf5:		{body_pose_hdf5[:, :6]}")
print(f"body_pose_pickled:	{body_pose_pickled[:, :6]}", end='\n\n')

print(f"transl_hdf5:		{transl_hdf5_meter_adj}")
print(f"transl_pickled:		{transl_pickled_meter}", end='\n\n')

print(f"___ SMPL Root Joint Position/Translation Comparison __", end='\n\n')
print(f"transl_hdf5_meter:		{transl_hdf5_meter}")
print(f"transl_pickled_meter:		{transl_pickled_meter}", end='\n\n')
print(f"transl_hdf5_meter_adj:		{transl_hdf5_meter_adj}")
print(f"transl_pickled_meter_adj:	{transl_pickled_meter_adj}")

# 4.4. Hence proved all the 4 SMPL parameters and joint positions are same in both HDF5 and Pickle files



### 5. Multiple SMPL Forward Passes

# 5.1. transl = meter (not adjusted)
smpl_output = model(body_pose=body_pose_pickled, global_orient=global_orient_pickled, betas=betas_pickled, transl=transl_pickled_meter)
smpl_markers_meter = smpl_output.joints.detach().cpu().numpy().squeeze()
print(f"smpl_markers_meter:	{smpl_markers_meter.shape}")

# 5.2. transl = meter (adjusted)
smpl_output_adj = model(body_pose=body_pose_pickled, global_orient=global_orient_pickled, betas=betas_pickled, transl=transl_pickled_meter_adj)
smpl_markers_meter_adj = smpl_output_adj.joints.detach().cpu().numpy().squeeze()
print(f"smpl_markers_meter_adj:	{smpl_markers_meter_adj.shape}")



### 6. Visualize SMPL Mesh with Joints, using Trimesh and Pyrender

## 6.1. Parameters & Joints loaded from Pickle file (Mod=1)

# # 6.1.1. joints = meter (not adjusted) | transl = meter (not adjusted)
# visualize_smpl_with_joints(
#     model			= model,
#     body_pose_1     = body_pose_pickled,
#     global_orient_1 = global_orient_pickled,
#     betas_1         = betas_pickled,
#     transl_1        = transl_pickled_meter,
#     joints_1        = joint_positions_pickled_meter[0],
#     show_ground     = False,
# )

# # 6.1.2. joints = meter (adjusted) | transl = meter (adjusted)
# visualize_smpl_with_joints(
#     model           = model,
#     body_pose_1     = body_pose_pickled,
#     global_orient_1 = global_orient_pickled,
#     betas_1         = betas_pickled,
#     transl_1        = transl_pickled_meter_adj,
#     joints_1        = joint_positions_pickled_meter_adj[0],
#     show_ground     = False,
# )

# 6.1.3. joints = meter (not adjusted) | transl = meter (calculated adjusted)
visualize_smpl_with_joints(
	model           = model,
	body_pose_1     = body_pose_pickled,
	global_orient_1 = global_orient_pickled,
	betas_1         = betas_pickled,
	transl_1        = transl_pickled_meter_adj_calc,
	joints_1        = joint_positions_pickled_meter[0],
	show_ground     = False,
	joint_radius    = 0.05,  # Increased from 0.015 to 0.05 for better visibility
)


## 6.2. Parameters loaded from Pickle file (Mod=1) & Joints returned by SMPL model

# # 6.2.1. joints = meter (not adjusted) | transl = meter (not adjusted)
# visualize_smpl_with_joints(
#     model           = model,
#     body_pose_1     = body_pose_pickled,
#     global_orient_1 = global_orient_pickled,
#     betas_1         = betas_pickled,
#     transl_1        = transl_pickled_meter,
#     joints_1        = smpl_markers_meter[:24],
#     show_ground     = False,
#     joint_radius    = 0.05,  # Increased from 0.015 to 0.05 for better visibility
# )

# # 6.2.2. joints = meter (adjusted) | transl = meter (adjusted)
# visualize_smpl_with_joints(
#     model           = model,
#     body_pose_1     = body_pose_pickled,
#     global_orient_1 = global_orient_pickled,
#     betas_1         = betas_pickled,
#     transl_1        = transl_pickled_meter_adj,
#     joints_1        = smpl_markers_meter_adj,
#     show_ground     = False,
# )

# # 6.2.3. joints = meter (not adjusted) | transl = meter (adjusted)
# visualize_smpl_with_joints(
#     model           = model,
#     body_pose_1     = body_pose_pickled,
#     global_orient_1 = global_orient_pickled,
#     betas_1         = betas_pickled,
#     transl_1        = transl_pickled_meter_adj,
#     joints_1        = smpl_markers_meter[:24],
#     show_ground     = False,
# )



### 7. Compare (by printing) the Ground Truth (from .p) and SMPL Forward Pass Joints
NUM_JOINTS = 24

# Convert from meter to cm for better visibility
smpl_joints_cm = smpl_markers_meter * 100
smpl_joints_cm_adj = smpl_markers_meter_adj * 100
pickle_joints_cm = joint_positions_pickled_meter[0] * 100
pickle_joints_cm_corrected = pickle_joints_cm + np.array([0.03, -20.48, 3.39])
										# x[:, 10:13] += [0.6, 1.2, 0.1]
# [0.03, -20.48, 3.39]cm = [0.3, -204.8, 33.9]mm = [0.0003, -0.2048, 0.0339]m

print(f"smpl_joints_cm:			{smpl_joints_cm.shape}")
print(smpl_joints_cm[:NUM_JOINTS])
print(f"smpl_joints_cm_adj:		{smpl_joints_cm_adj.shape}")
print(smpl_joints_cm_adj[:NUM_JOINTS])
print(f"pickle_joints_cm_corrected:	{pickle_joints_cm_corrected.shape}")
print(pickle_joints_cm_corrected[:NUM_JOINTS])
print(f"pickle_joints_cm:		{pickle_joints_cm.shape}")
print(pickle_joints_cm[:NUM_JOINTS])

# diff = smpl_joints_cm[:24] - pickle_joints_cm_corrected[:24]
# print(f"Difference (SMPL - Pickle):	{diff.shape}")
# print(diff[:24])