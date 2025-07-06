import h5py
import numpy as np
import os
import random
from utils import get_preprocessed_hdf5_path

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Original file
input_file = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
input_path = get_preprocessed_hdf5_path(input_file)

# Output file
output_file = input_file.replace(".hdf5", "__with_val_and_test_split.hdf5")
output_path = os.path.join(os.path.dirname(input_path), output_file)


with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:

	# Copy full train set as-is
	src.copy('train', dst)

	# Split test -> val & test (per gender)
	for gender in ['f', 'm']:
		inputs = src[f'test/straight_limbs/{gender}/inputs'][:]
		labels = src[f'test/straight_limbs/{gender}/labels'][:]

		N = len(inputs)
		indices = np.arange(N)
		np.random.shuffle(indices)

		val_idx = indices[:300]
		test_idx = indices[300:]

		inputs_val = inputs[val_idx]
		labels_val = labels[val_idx]

		inputs_test = inputs[test_idx]
		labels_test = labels[test_idx]

		# Create group hierarchy
		for split_name, data in [('val', (inputs_val, labels_val)), ('test', (inputs_test, labels_test))]:
			grp = dst.require_group(f'{split_name}/straight_limbs/{gender}')
			grp.create_dataset('inputs', data=data[0], dtype='float32')
			grp.create_dataset('labels', data=data[1], dtype='float32')

print(f"✅ New HDF5 saved to: {output_path}")