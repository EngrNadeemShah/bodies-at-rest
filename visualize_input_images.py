### 0. Initialization

# Import libraries
import sys
import os
import torch
import numpy as np
import random
import pickle
import h5py

from torch.utils.data import DataLoader
from datasets import HDF5Dataset
from utils import plot_input_channels, retrieve_data_file_paths

np.set_printoptions(threshold=sys.maxsize, precision=2, suppress=True)

## Paths

# Path to hdf5 dataset
# hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'


# Pickle file paths
# crossed_legs (mod=1)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_f_lay_set2both_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/crossed_legs/train_roll0_xl_m_lay_set2both_4000.p'

# straight_limbs (mod=1)
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod1/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000.p'

# # straight_limbs (mod=2)
pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'
# pkl_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/original/mod2/train/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000_convnet_1_anglesDC_184000ct_128b_x1pm_tnh_100e_2e-05lr.p'

index = 0

# # Load the HDF5 file
# with h5py.File(hdf5_file_path, 'r') as f:
# 	# inputs_hdf5 = f['train/straight_limbs/f/inputs'][index]
# 	inputs_hdf5 = f['train']
# 	print(inputs_hdf5)
# 	# print(inputs_hdf5.shape)
# 	# plot_input_channels(inputs_hdf5)


with h5py.File(hdf5_file_path, 'r') as f:
    # List all groups in the file
    def print_hdf5_structure(g, prefix=''):
        for key in g:
            item = g[key]
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                print_hdf5_structure(item, prefix + '  ')
            else:
                print(f"{prefix}{key} - shape: {item.shape}, dtype: {item.dtype}")

    print_hdf5_structure(f)


# Load the pickle file
with open(pkl_file_path, 'rb') as f:
	data = pickle.load(f, encoding='latin1')  # Use 'latin1' for Python 2 compatibility


# # 3.4. Print the keys and shapes of the data loaded from the pickle file
# for i, (key, value) in enumerate(data.items(), start=1):
# 	print(f"({i:02}/{len(data):02}) {key}:\t{value.shape if isinstance(value, np.ndarray) else value.size() if isinstance(value, torch.Tensor) else np.array(value).shape if isinstance(value, list) else type(value)}")


# # 3.5. Fetch the required data from the pickle file
images		= torch.tensor(np.array(data['images'])).float()[index].unsqueeze(0).reshape(-1, 64, 27)	# dtype=int8
mesh_contact= torch.tensor(np.array(data['mesh_contact'])).float()[index].unsqueeze(0)				# dtype=bool
mesh_depth	= torch.tensor(np.array(data['mesh_depth'])).float()[index].unsqueeze(0)					# dtype=int32

input_pickled = torch.cat((images, mesh_contact, mesh_depth), dim=0).unsqueeze(0)
print(f"input_pickled:		{input_pickled.shape}")
# plot_input_channels(input_pickled)