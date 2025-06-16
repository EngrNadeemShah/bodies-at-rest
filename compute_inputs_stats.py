import h5py
import numpy as np
import pandas as pd

def calculate_stats(inputs):
	"""Calculate min, max, mean, std for each channel of inputs"""
	stats = {
		'min': np.min(inputs, axis=(0, 2, 3)),  # Assuming inputs shape is (batch_size, channels, height, width)
		'max': np.max(inputs, axis=(0, 2, 3)),
		'mean': np.mean(inputs, axis=(0, 2, 3)),
		'std_dev': np.std(inputs, axis=(0, 2, 3))
	}
	return stats

def process_hdf5_inputs(hdf5_path, output_csv_prefix):
	"""Process train inputs and save stats to CSV with multiple sheets"""
	with h5py.File(hdf5_path, 'r') as file:
		# Initialize Excel writer
		writer = pd.ExcelWriter(f'stats_{output_csv_prefix}_inputs_processed.xlsx', engine='xlsxwriter')

		# Process each group (train or test)
		for split in ['train']:
			if split in file:
				split_group = file[split]
				for pose in split_group:
					pose_group = split_group[pose]
					for gender in ['f', 'm']:
						if gender in pose_group:
							# Get inputs
							inputs = pose_group[gender]['inputs'][:]

							# Calculate stats
							stats = calculate_stats(inputs)

							# Create DataFrame
							df = pd.DataFrame(stats)
							df.index = [f'channel_{i}' for i in range(inputs.shape[1])]

							# Save to sheet
							sheet_name = f"{gender}_{pose}"
							if len(sheet_name) > 31:  # Excel sheet name limit
								sheet_name = sheet_name[:31]
							df.to_excel(writer, sheet_name=sheet_name)

							print(f"Processed {sheet_name} with shape {inputs.shape}")

		# Save the Excel file
		writer.close()

# HDF5 file paths
hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nashah/scratch/data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'

# hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
# hdf5_file_path = '/home/nadeemshah/coding/bodies-at-rest/synthetic_data/pre_processed/preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'


# Process train and test labels
process_hdf5_inputs(hdf5_file_path, 'train')

print("Statistics saved to stats_train_inputs_processed.xlsx")