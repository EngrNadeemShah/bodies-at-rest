"""__!!! Activate smplx/.conda environment before running this notebook.__  
# Won't run on Headless server (e.g. HPC) because it requires GUI."""


### 1. Initialization

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

from torch.utils.data import DataLoader
from datasets import HDF5Dataset
from utils import visualize_smpl_with_joints, get_preprocessed_hdf5_path

np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)

# HDF5 file paths
# hdf5_file_name = 'preprocessed_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_name = 'preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'
# hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True_no_75mm.hdf5'
# hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True__with_val_and_test_split.hdf5'

hdf5_file_path = get_preprocessed_hdf5_path(hdf5_file_name)

# Paths to SMPL models
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

# DataLoader setup
batch_size = 1
train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train', verbose=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


### 2. Set paths and parameters

# Adjust z-coordinate of the root joint (pelvis) by this amount to correct for elevation offset from FleX simulation settling
z_adj = -0.075		# -0.075 m = -7.5 cm (-ve value means downwards shift)
# z_adj = -0.75

# Path to the pickle files containing SMPL parameters
# crossed_legs (mod=1)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_f_lay_set2both_4000.p'
pkl_file_path = '/home/nadeemshah/scratch/data/original/mod1/train/crossed_legs/train_roll0_xl_f_lay_set2both_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_m_lay_set2both_4000.p'

# straight_limbs (mod=1)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000.p'

# # straight_limbs (mod=2)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'

# Get filename and directory information
filename = os.path.basename(pkl_file_path)
filename_splitted = filename.split('_')
dirname = os.path.dirname(pkl_file_path)
dirname_splitted = dirname.split('/')
pose_type = dirname_splitted[-1]	# 'crossed_legs', 'general', 'general_supine', 'hands_behind_head', 'prone_hands_up', 'straight_limbs'
mod = dirname_splitted[-3]			# 'mod1', 'mod2'

# Robustly fetch the number of examples from the splitted filename
num_examples = None
for part in filename_splitted:
	# Look for a part that is all digits (e.g., '4000', '10000', etc.)
	if part.isdigit():
		num_examples = int(part)
		break

if num_examples is None:
    num_examples = filename_splitted[-1].split('.')[0]  # Fallback to the last part before the file extension
    num_examples = int(num_examples) if num_examples.isdigit() else None

if num_examples is None:
	raise ValueError("Could not determine the number of examples from the filename.")

# Determine the gender type based on the filename
if filename_splitted.__contains__('m'):
	gender_type = 'male'
elif filename_splitted.__contains__('f'):
    gender_type = 'female'
else:
    gender_type = None

# Set a random index to visualize a specific example
# random_index = random.randint(0, num_examples - 1)
random_index = 0

# Print the extracted information
print(f"dirname:	{dirname}")
print(f"filename:	{filename}")
print(f"pose_type:	{pose_type}")
print(f"mod:		{mod}")
print(f"gender_type:	{gender_type}")
print(f"random_index:	{random_index}")
print(f"num_examples:	{num_examples}")
print(f"z_adj:		{z_adj}")


### 3. Load SMPL parameters

# Load the pickle file containing SMPL parameters
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # Use 'latin1' for Python 2 compatibility

# Print the keys and shapes of the data loaded from the pickle file
# for key, value in data.items():
# 	print(f"{key}:\t {value.shape if isinstance(value, np.ndarray) else value.size() if isinstance(value, torch.Tensor) else np.array(value).shape if isinstance(value, list) else type(value)}")

# Load different SMPL parameters based on the mod type
# Mod=1
betas_orig			= torch.tensor(np.array(data['body_shape'])).float()[random_index].unsqueeze(0)				# Shape: (batch_size, 10)
global_orient_orig	= torch.tensor(np.array(data['joint_angles'])).float()[:, 0:3][random_index].unsqueeze(0)	# Shape: (batch_size, 3)
body_pose_orig		= torch.tensor(np.array(data['joint_angles'])).float()[:, 3:72][random_index].unsqueeze(0)	# Shape: (batch_size, 69)
transl_orig			= torch.tensor(np.array(data['root_xyz_shift'])).float()[random_index].unsqueeze(0)			# Shape: (batch_size, 3)
# Shift root joint (pelvis) z down (-ve) or up (+ve) by z_adj to correct for elevation offset from FleX simulation settling
transl_z_adj_orig	= torch.tensor(np.array(data['root_xyz_shift']) + np.array([0, 0, z_adj])).float()[random_index].unsqueeze(0)

# # Mod=2
# betas_est			= torch.tensor(np.array(data['betas_est'])).float()[random_index].unsqueeze(0)				# Shape: (batch_size, 10)
# global_orient_est   = torch.tensor(np.array(data['angles_est'])).float()[:, 0:3][random_index].unsqueeze(0)		# Shape: (batch_size, 3)
# body_pose_est       = torch.tensor(np.array(data['angles_est'])).float()[:, 3:72][random_index].unsqueeze(0)	# Shape: (batch_size, 69)
# transl_est          = torch.tensor(np.array(data['root_xyz_est'])).float()[random_index].unsqueeze(0)			# Shape: (batch_size, 3)
# # Shift root joint (pelvis) z down (-ve) or up (+ve) by z_adj to correct for elevation offset from FleX simulation settling
# transl_z_adj_est	= torch.tensor(np.array(data['root_xyz_est']) + np.array([0, 0, z_adj])).float()[random_index].unsqueeze(0)

# Loaded from HDF5 dataset (Mod=1)
for inputs, true_labels in train_loader:
	is_female = true_labels[:, 157].bool()

	# Extract SMPL parameters from true_labels
	betas_hdf5           = true_labels[:, 72:82]     # Shape: (batch_size, 10)
	global_orient_hdf5   = true_labels[:, 82:85]     # Shape: (batch_size, 3)
	body_pose_hdf5       = true_labels[:, 85:154]    # Shape: (batch_size, 69)
	transl_hdf5          = true_labels[:, 154:157]   # Shape: (batch_size, 3)

	# Print the shapes of the loaded SMPL parameters
	# print(f"Batch Size:	{inputs.shape[0]}")
	# print(f"Is Female:	{is_female}")
	# print(f"Betas:		{np.array(betas)}")
	# print(f"Global Orient:	{np.array(global_orient)}")
	# print(f"Body Pose:	{np.array(body_pose).shape}")
	# print(f"Transl:		{np.array(transl)}")
	break


### 4. Visualize SMPL parameters

### 4.1. Single human model visualization
# # Original (without z)
# visualize_smpl_with_joints(
# 	model=model, body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_orig)

# # Estimated (without z)
# visualize_smpl_with_joints(
# 	model=model, body_pose_1=body_pose_est, global_orient_1=global_orient_est, betas_1=betas_est, transl_1=transl_est)

# # Original (with z)
# visualize_smpl_with_joints(
# 	model=model, body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig)

# # Estimated (with z)
# visualize_smpl_with_joints(
# 	model=model, body_pose_1=body_pose_est, global_orient_1=global_orient_est, betas_1=betas_est, transl_1=transl_z_adj_est)


### 4.2. Human Model 1 (Blue) | Human Model 2 (Purple)

# # Original (without z) | Estimated (without z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_orig,
# 	body_pose_2=body_pose_est, global_orient_2=global_orient_est, betas_2=betas_est, transl_2=transl_est)

# # Original (with z) | Estimated (with z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig,
# 	body_pose_2=body_pose_est, global_orient_2=global_orient_est, betas_2=betas_est, transl_2=transl_z_adj_est)

# # Original (without z) | Original (with z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_orig,
# 	body_pose_2=body_pose_orig, global_orient_2=global_orient_orig, betas_2=betas_orig, transl_2=transl_z_adj_orig)

# # Estimated (without z) | Estimated (with z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_est, global_orient_1=global_orient_est, betas_1=betas_est, transl_1=transl_est,
# 	body_pose_2=body_pose_est, global_orient_2=global_orient_est, betas_2=betas_est, transl_2=transl_z_adj_est)

# # Original (without z) | Estimated (with z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_orig,
# 	body_pose_2=body_pose_est, global_orient_2=global_orient_est, betas_2=betas_est, transl_2=transl_z_adj_est)

# # Original (with z) | Estimated (without z)	-> this is the correct one (as estimated_smpl_params are the output of the trained mod1, which was trained on original z-adjusted ground truth, therefore, the estimated parameters are already adjusted for z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig,
# 	body_pose_2=body_pose_est, global_orient_2=global_orient_est, betas_2=betas_est, transl_2=transl_est)


### 4.3. Raw (.p) vs Processed (.hdf5) dataset

# # Original (without z) | HDF5 (without z)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_orig,
# 	body_pose_2=body_pose_hdf5, global_orient_2=global_orient_hdf5, betas_2=betas_hdf5, transl_2=transl_hdf5)

# # Original (with z) | HDF5 (with z=z_adj)
# transl_z_adj_hdf5 = transl_hdf5 + torch.tensor([0, 0, z_adj]).float().unsqueeze(0)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig,
# 	body_pose_2=body_pose_hdf5, global_orient_2=global_orient_hdf5, betas_2=betas_hdf5, transl_2=transl_z_adj_hdf5)

# Original (with z) | HDF5 (with z=infinitely small)
transl_z_adj_hdf5 = transl_hdf5 + torch.tensor([0, 0, 1e-5]).float().unsqueeze(0)  # Adding a very small value to z
visualize_smpl_with_joints(
	model=model,
	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig,
	body_pose_2=body_pose_hdf5, global_orient_2=global_orient_hdf5, betas_2=betas_hdf5, transl_2=transl_z_adj_hdf5)

# # Original (with z) | HDF5 (without z)	-> they both are the same, as HDF5 dataset is pre-processed using preprocess_data.py which already adjusts the z-coordinate of the root joint (pelvis) by z_adj=-0.075m (-7.5 cm)
# visualize_smpl_with_joints(
# 	model=model,
# 	body_pose_1=body_pose_orig, global_orient_1=global_orient_orig, betas_1=betas_orig, transl_1=transl_z_adj_orig,
# 	body_pose_2=body_pose_hdf5, global_orient_2=global_orient_hdf5, betas_2=betas_hdf5, transl_2=transl_hdf5)