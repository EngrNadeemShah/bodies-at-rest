import os
import pickle
import numpy as np
from scipy.ndimage import sobel, zoom, gaussian_filter
from random import normalvariate
from utils import retrieve_data_file_paths
from time import time
from datetime import datetime, timedelta
import psutil
import gc
import torch
import h5py


def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f, encoding='latin1')

def get_memory_usage():
    """Returns the memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024) # in GB


class DataPreprocessor:
	def __init__(self, config):
		self.config = config

	def save_preprocessed_data(self, data_dict, source_path):
		"""Save preprocessed data efficiently as PyTorch tensors."""

		# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		config_summary = '__'.join([f'{key}_{value}' for key, value in self.config.items() if key != 'normalize_std_dev'])
		source_path_parts = source_path.split('/')
		destination_path = os.path.join(source_path_parts[0], 'pre_processed', source_path_parts[2], config_summary, *source_path_parts[3:]).replace('.p', '.pt')
		os.makedirs(os.path.dirname(destination_path), exist_ok=True)

		# Convert numpy arrays to torch tensors before saving
		for key in data_dict:
			if isinstance(data_dict[key], np.ndarray):
				data_dict[key] = torch.from_numpy(data_dict[key])

		torch.save(data_dict, destination_path)
		print(f"Saved to {destination_path}")

	def save_data_as_hdf5(self, inputs, labels, source_pkl_path):
		"""
		Save preprocessed data efficiently into an HDF5 file, appending if the dataset already exists.

		Arguments:
		- inputs (np.array): Pressure map data.
		- labels (np.array): Corresponding labels.
		- source_path (str): Original file path used to infer hierarchy.
		"""

		config_summary = '__'.join([f'{key}_{value}' for key, value in self.config.items() if key != 'normalize_std_dev' and key != 'hdf5_file_path'])
		hdf5_file_path = self.config['hdf5_file_path']
		hdf5_file_path = hdf5_file_path.replace('.hdf5', f'_{config_summary}.hdf5')
		# Extract metadata from source_path
		parts = source_pkl_path.split(os.sep)
		pose_name = parts[-2]
		file_name = os.path.basename(source_pkl_path)
		split = "train" if "train" in file_name else "test" if "test" in file_name else None
		gender = "f" if "_f_" in file_name else "m" if "_m_" in file_name else None

		if split is None or gender is None:
			raise ValueError(f"Invalid source_path format: {source_pkl_path}")

		# Define hierarchical storage path
		group_path = f"{split}/{pose_name}/{gender}"

		# Open HDF5 file in append mode
		with h5py.File(hdf5_file_path, "a") as hdf5_file:
			# Create group if not exists
			if group_path not in hdf5_file:
				hdf5_file.create_group(group_path)

			# Function to append data safely
			def append_data(dataset_name, new_data):
				full_path = f"{group_path}/{dataset_name}"
				if full_path in hdf5_file:
					# Dataset exists → Extend dataset
					dset = hdf5_file[full_path]
					old_size = dset.shape[0]
					new_size = old_size + new_data.shape[0]
					dset.resize((new_size,) + dset.shape[1:])  # Expand first dimension
					dset[old_size:new_size] = new_data  # Append new data
				else:
					# Create dataset with chunking and compression
					max_shape = (None,) + new_data.shape[1:]  # Allow unlimited growth along first dimension
					hdf5_file.create_dataset(name=full_path, data=new_data, maxshape=max_shape, chunks=True, compression="gzip", compression_opts=9)

			# Append or create datasets
			append_data("inputs", inputs)
			append_data("labels", labels)

		print(f"HDF5 data appended under: {group_path}")

	def process_pressure_map(self, pressure_maps):
		pressure_maps = np.clip(pressure_maps, 0, 100).reshape(-1, 64, 27)
		if self.config['add_noise'] == 0:
			pressure_maps = gaussian_filter(pressure_maps, sigma=(0, 0.5, 0.5))
		pressure_maps = np.clip(pressure_maps, 0, 100)
		if self.config['normalize_per_image']:
			pressure_maps = pressure_maps * 20000.0 / pressure_maps.sum(axis=(1, 2), keepdims=True)
		return pressure_maps

	def apply_sobel_filter(self, pressure_maps):
		sx = sobel(pressure_maps, axis=1, mode='constant')	# Apply along height (64)
		sy = sobel(pressure_maps, axis=2, mode='constant')	# Apply along width (27)
		# Compute gradient magnitude (hypotenuse)
		pm_sobel_filtered = np.hypot(sx, sy)

		if self.config['add_noise'] == 0:
			pm_sobel_filtered = np.clip(pm_sobel_filtered, 0, 100)
		if self.config['normalize_per_image']:
			pm_sobel_filtered *= 20000.0 / pm_sobel_filtered.sum(axis=(1, 2), keepdims=True)
		if self.config['add_noise'] != 0:
			pm_sobel_filtered *= np.random.normal(loc=1.0, scale=self.config['add_noise'], size=(pm_sobel_filtered.shape[0], 1, 1))

		pm_sobel_filtered *= self.config['normalize_std_dev'][5]
		if self.is_train:
			pm_sobel_filtered = self.apply_noise_into_training(pm_sobel_filtered, channel_idx=5)
		return pm_sobel_filtered

	def apply_noise_into_training(self, images, channel_idx):
		# Generate Gaussian-distributed noise directly
		noise = np.random.normal(loc=0, scale=1, size=images.shape)  
		# Apply noise only to nonzero pixels
		images += noise * (images != 0) * self.config['normalize_std_dev'][channel_idx]
		# Clip values to valid range
		return np.clip(images, 0, 10000)

	def apply_noise_to_pressure_map(self, pressure_maps):
		if self.is_train:
			pressure_map_threshold = (pressure_maps != 0).astype(np.float32)

			noise_multiplier = np.random.normal(loc=1.0, scale=self.config['add_noise'], size=(pressure_maps.shape[0], 1, 1))
			noise_addition = np.random.normal(
				loc=0.0, 
				scale=self.config['normalize_std_dev'][4] * 
						(70.0 if self.config['normalize_per_image'] else 98.666) * 
						self.config['add_noise'],
				size=(pressure_maps.shape[0], 1, 1))

			pressure_maps *= noise_multiplier
			pressure_maps += noise_addition

			pressure_maps = np.clip(pressure_maps, 0, 10000)
			pressure_maps *= pressure_map_threshold

					# Apply Gaussian filter to the pressure maps
			sigma_values = np.random.normal(loc=0.5, scale=self.config['add_noise'], size=pressure_maps.shape[0])
			pressure_maps = np.stack([gaussian_filter(pm, sigma=sigma) for pm, sigma in zip(pressure_maps, sigma_values)])
		else:
			pressure_maps = gaussian_filter(pressure_maps, sigma=(0, 0.5, 0.5))
			pressure_maps = np.clip(pressure_maps, 0, 10000 if self.config['normalize_per_image'] else 100 * self.config['normalize_std_dev'][4])
		return pressure_maps

	def create_contact_map(self, pressure_maps):
		return np.where(pressure_maps != 0, 100.0 * self.config['normalize_std_dev'][0], 0)

	def include_estimated_depth_contact_maps(self, file_data):
		depth_map_estimated = np.array(file_data['mdm_est'])
		depth_map_estimated_positive = np.maximum(depth_map_estimated, 0)
		depth_map_estimated_negative = np.maximum(-depth_map_estimated, 0)
		contact_map_estimated = np.array(file_data['cm_est']) * 100.0
		depth_map_estimated_positive *= self.config['normalize_std_dev'][1]
		depth_map_estimated_negative *= self.config['normalize_std_dev'][2]
		contact_map_estimated *= self.config['normalize_std_dev'][3]
		if self.config['use_hover']:
			depth_map_estimated_positive = np.zeros_like(depth_map_estimated_positive)
		return depth_map_estimated_positive, depth_map_estimated_negative, contact_map_estimated

	def include_depth_contact_maps(self, file_data):
		depth_map = np.array(file_data['mesh_depth']).astype(np.float32)
		contact_map = np.array(file_data['mesh_contact']).astype(np.float32)
		depth_map *= self.config['normalize_std_dev'][6]
		contact_map *= self.config['normalize_std_dev'][7]
		return depth_map, contact_map

	def include_height_weight(self, file_data):
		weight = np.array(file_data['body_mass']) * self.config['normalize_std_dev'][8]
		weight_channel = np.full(shape=(len(file_data['body_mass']), 64, 27), fill_value=weight[:, None, None])
		height = (np.array(file_data['body_height']) - 1.0) * 100 * self.config['normalize_std_dev'][9]
		height_channel = np.full(shape=(len(file_data['body_height']), 64, 27), fill_value=height[:, None, None])
		return weight_channel, height_channel

	def concatenate_input_channels(self, file_data, pm_contact_map, pressure_maps, pm_sobel_filtered):
		input_x = np.zeros((len(pressure_maps), 0, 64, 27), dtype=np.float32)

		if not self.config['omit_contact_sobel']:
			input_x = np.concatenate((input_x, pm_contact_map[:, None, :, :]), axis=1)

		if self.config['mod'] == 2:
			depth_map_estimated_positive, depth_map_estimated_negative, contact_map_estimated = self.include_estimated_depth_contact_maps(file_data)
			input_x = np.concatenate((input_x, depth_map_estimated_positive[:, None, :, :], depth_map_estimated_negative[:, None, :, :], contact_map_estimated[:, None, :, :]), axis=1)

		if self.config['omit_contact_sobel']:
			input_x = np.concatenate((input_x, pressure_maps[:, None, :, :]), axis=1)
		else:
			input_x = np.concatenate((input_x, pressure_maps[:, None, :, :], pm_sobel_filtered[:, None, :, :]), axis=1)

		if self.config['mod'] == 2:
			depth_map, contact_map = self.include_depth_contact_maps(file_data)
			input_x = np.concatenate((input_x, depth_map[:, None, :, :], contact_map[:, None, :, :]), axis=1)

		if self.config['include_weight_height']:
			weight_channel, height_channel = self.include_height_weight(file_data)
			input_x = np.concatenate((input_x, weight_channel[:, None, :, :], height_channel[:, None, :, :]), axis=1)

		return input_x

	def load_input(self, file_data):
		pressure_maps = file_data['images']
		pressure_maps = self.process_pressure_map(pressure_maps)

		pm_sobel_filtered = self.apply_sobel_filter(pressure_maps)

		pressure_maps *= self.config['normalize_std_dev'][4]

		if self.config['add_noise'] != 0:
			pressure_maps = self.apply_noise_to_pressure_map(pressure_maps)

		pm_contact_map = self.create_contact_map(pressure_maps)
	
		if self.is_train:
			pressure_maps = self.apply_noise_into_training(pressure_maps, channel_idx=4)

		inputs = self.concatenate_input_channels(file_data, pm_contact_map, pressure_maps, pm_sobel_filtered)

		# Upsample input channels by a factor of 2 (64x27 -> 128x54)
		inputs = zoom(inputs, zoom=(1, 1, 2, 2), order=1)
		return inputs

	def load_label(self, file_data, g1, g2):
		s1 = np.ones_like(g1)	# always 1 because we always have synthetic data

		z_adj = -0.075
		z_adj_all = np.array(24 * [0, 0, z_adj * 1000], dtype=np.float32)

		labels = np.concatenate([
			np.array(file_data['markers_xyz_m'], dtype=np.float32) * 1000 + z_adj_all,									# 0:72		(72)
			np.array(file_data['body_shape'], dtype=np.float32),														# 72:82		(10)
			np.array(file_data['joint_angles'], dtype=np.float32),														# 82:154	(72)
			np.array(file_data['root_xyz_shift'], dtype=np.float32) + np.array([0, 0, z_adj], dtype=np.float32),		# 154:157	(3)
			g1[:, None], g2[:, None], s1[:, None],																		# 157:160	(3)
			np.array(file_data['body_mass'], dtype=np.float32)[:, None] * self.config['normalize_std_dev'][8],					# 160:161	(1)
			(np.array(file_data['body_height'], dtype=np.float32)[:, None] - 1.0) * 100 * self.config['normalize_std_dev'][9]	# 161:162	(1)
		], axis=1)

		if self.config['mod'] == 2:
			labels = np.concatenate([
				labels,
				np.array(file_data['betas_est'], dtype=np.float32),			# 162:172	(10)
				np.array(file_data['angles_est'], dtype=np.float32),		# 172:244	(72)
				np.array(file_data['root_xyz_est'], dtype=np.float32),		# 244:247	(3)
				np.array(file_data['root_atan2_est'], dtype=np.float32)		# 247:253	(6)
			], axis=1)
		return labels

	def preprocess_single_file(self, file_path):
		file_data = load_pickle(file_path)
		g1 = np.array([1 if '_f_' in file_path else 0] * len(file_data['images']))
		g2 = np.array([1 if '_m_' in file_path else 0] * len(file_data['images']))

		# Process all data indices at once
		inputs = self.load_input(file_data).astype(np.float32)
		labels = self.load_label(file_data, g1, g2).astype(np.float32)

		print(f'Inputs: {inputs.shape} | {inputs.dtype}')
		print(f'Labels: {labels.shape} | {labels.dtype}')

		# Save the preprocessed data
		# data_dict = {'inputs': inputs, 'labels': labels}
		# self.save_preprocessed_data(data_dict, file_path)

		# Save the preprocessed data using HDF5 storage
		self.save_data_as_hdf5(inputs, labels, source_pkl_path=file_path)

		# Free memory explicitly
		del inputs, labels, file_data	# , data_dict
		gc.collect()

	def preprocess_data(self, file_paths, is_train=True):
		self.is_train = is_train
		for i, file_path in enumerate(file_paths):
			# Measure time and memory usage
			single_file_time = time()
			mem_before = get_memory_usage()

			print(f'File ({i + 1:02}/{len(file_paths):02}): {file_path}')
			self.preprocess_single_file(file_path)
			print(f'Time taken: {time() - single_file_time:.2f} seconds')
			print(f'Memory usage: {mem_before:.2f} GB -> {get_memory_usage():.2f} GB \n')


if __name__ == '__main__':
	start_time = time()
	config = {
		'add_noise':				0,
		'include_weight_height':	False,
		'omit_contact_sobel':		False,
		'use_hover':				False,
		'mod':						1,		# 1 or 2
		'normalize_per_image':		True,
		'hdf5_file_path':			f'synthetic_data/pre_processed/preprocessed.hdf5'
	}

	config['hdf5_file_path'] = config['hdf5_file_path'].replace('.hdf5', f'_mod{config["mod"]}.hdf5')

	# Normalization standard deviations for each channel
	if config['normalize_per_image']:
		config['normalize_std_dev'] = np.ones(10, dtype=np.float32)
	else:
		config['normalize_std_dev'] = np.array([
			1 / 41.8068,	# 0 contact
			1 / 16.6955,	# 1 pos est depth
			1 / 45.0851,	# 2 neg est depth
			1 / 43.5580,	# 3 cm est
			1 / 11.7015,	# 4 pressure map
			1 / 45.6164 if config['add_noise'] else 1 / 29.8036,	# 5 pressure map sobel
			1,  # 6 OUTPUT DO NOTHING
			1,  # 7 OUTPUT DO NOTHING
			1 / 30.2166,  # 8 weight
			1 / 14.6293   # 9 height
		])

	# Define the path to the original train and test data (.pickle files)
	if config['mod'] == 1:
		train_files_dir = 'synthetic_data/original/mod1/train'
		valid_files_dir = 'synthetic_data/original/mod1/test'
	elif config['mod'] == 2:
		train_files_dir = 'synthetic_data/original/mod2/train'
		valid_files_dir = 'synthetic_data/original/mod2/test'

	# Get the paths to the train and test data files
	train_file_paths = retrieve_data_file_paths(train_files_dir)
	valid_file_paths = retrieve_data_file_paths(valid_files_dir)

	# Create a DataPreprocessor instance
	preprocessor = DataPreprocessor(config)

	# Preprocess the train and test data
	preprocessor.preprocess_data(train_file_paths, is_train=True)
	preprocessor.preprocess_data(valid_file_paths, is_train=False)
	total_time = time() - start_time
	formatted_time = str(timedelta(seconds=int(total_time)))
	print(f'Total execution time: {formatted_time}')