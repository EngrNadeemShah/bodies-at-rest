import h5py
import numpy as np
import pandas as pd

def calculate_stats(labels):
	"""Calculate min, max, mean, std for each axis of labels"""
	stats = {
		'min': np.min(labels, axis=0),
		'max': np.max(labels, axis=0),
		'mean': np.mean(labels, axis=0),
		'std_dev': np.std(labels, axis=0)
	}
	return stats

def process_hdf5_labels(hdf5_path, output_csv_prefix):
	"""Process train labels and save stats to CSV with multiple sheets"""
	with h5py.File(hdf5_path, 'r') as file:
		# Initialize Excel writer
		writer = pd.ExcelWriter(f'stats_{output_csv_prefix}_labels_processed.xlsx', engine='xlsxwriter')

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
hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'

# hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'


# Process train and test labels
process_hdf5_labels(hdf5_file_path, 'train')

print("Statistics saved to stats_train_labels_processed.xlsx")