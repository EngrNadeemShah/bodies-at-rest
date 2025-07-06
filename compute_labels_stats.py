import h5py
import numpy as np
import pandas as pd
from utils import get_preprocessed_hdf5_path

def calculate_stats(labels):
	"""Calculate min, max, mean, std for each axis of labels"""
	stats = {
		'min': np.min(labels, axis=0),
		'max': np.max(labels, axis=0),
		'mean': np.mean(labels, axis=0),
		'std_dev': np.std(labels, axis=0)
	}
	return stats

def process_hdf5_labels(hdf5_path, output_filename='stats_train_labels_processed.xlsx'):
	"""Process train labels and save stats to CSV with multiple sheets"""
	with h5py.File(hdf5_path, 'r') as file:
		# Initialize Excel writer
		writer = pd.ExcelWriter(output_filename, engine='openpyxl')

		# Process each group (train or test)
		for split in ['train']:
			if split in file:
				split_group = file[split]
				for pose in split_group:
					pose_group = split_group[pose]
					for gender in ['f', 'm']:
						if gender in pose_group:
							# Get labels
							labels = pose_group[gender]['labels'][:]

							# Calculate stats
							stats = calculate_stats(labels)

							# Create DataFrame
							df = pd.DataFrame(stats)
							df.index = [f'axis_{i}' for i in range(labels.shape[1])]

							# Save to sheet
							sheet_name = f"{gender}_{pose}"
							if len(sheet_name) > 31:  # Excel sheet name limit
								sheet_name = sheet_name[:31]
							df.to_excel(writer, sheet_name=sheet_name)

							print(f"Processed {sheet_name} with shape {labels.shape}")

		# Save the Excel file
		writer.close()

# HDF5 file paths
# hdf5_file_name = 'preprocessed_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_name = 'preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'
# hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True_no_75mm.hdf5'
hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

hdf5_file_path = get_preprocessed_hdf5_path(hdf5_file_name)

# Process train and test labels
output_filename = 'stats_train_labels_processed_straight_limbs.xlsx'
process_hdf5_labels(hdf5_file_path, output_filename)

print(f"Statistics saved to {output_filename}")