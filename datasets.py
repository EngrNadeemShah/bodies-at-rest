import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel, zoom
from random import normalvariate
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
mpl.rcParams['text.usetex'] = False  # Disable LaTeX rendering
mpl.rcParams['font.family'] = 'DejaVu Sans'  # Set default font family
mpl.use('Agg')
# mpl.use('TkAgg')

def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f, encoding='latin1')

class PressurePoseDataset(Dataset):
	def __init__(self, file_paths, config, is_train: bool):
		self.file_paths = file_paths
		self.config = config
		self.is_train = is_train
		self.data_indices = self._index_data()

	def _index_data(self):
		"""
		Index the data across all files without loading it into memory.
		Returns a list of tuples (file_path, data_idx, gender).
		"""
		data_indices = []
		for file_path in self.file_paths:
			# Identify the gender from the file path
			gender = 'f' if '_f_' in file_path else 'm' if '_m_' in file_path else None

			# Load the pickle file
			file_data = load_pickle(file_path)

			# Index the data based on the key 'images'
			for data_idx in range(len(file_data['images'])):
				data_indices.append((file_path, data_idx, gender))

		return data_indices

	def __len__(self):
		return len(self.data_indices)

	def _process_pressure_map(self, pressure_map):
		pressure_map = np.clip(pressure_map, 0, 100).reshape(64, 27)
		if self.config['add_noise'] == 0:
			pressure_map = gaussian_filter(pressure_map, sigma=0.5)
		pressure_map = np.clip(pressure_map, 0, 100)
		if self.config['normalize_per_image']:
			pressure_map = pressure_map * 20000.0 / pressure_map.sum()
		return pressure_map

	def _apply_sobel_filter(self, pressure_map):
		sx = sobel(pressure_map, axis=0, mode='constant')
		sy = sobel(pressure_map, axis=1, mode='constant')
		pm_sobel_filtered = np.hypot(sx, sy)

		if self.config['add_noise'] == 0:
			pm_sobel_filtered = np.clip(pm_sobel_filtered, 0, 100)
		if self.config['normalize_per_image']:
			pm_sobel_filtered *= 20000.0 / pm_sobel_filtered.sum()
		if self.config['add_noise'] != 0:
			pm_sobel_filtered *= normalvariate(mu=1.0, sigma=self.config['add_noise'])
		pm_sobel_filtered *= self.config['normalize_std_dev'][5]
		if self.is_train:
			pm_sobel_filtered = self._apply_noise_into_training(pm_sobel_filtered, channel_idx=5)
		return pm_sobel_filtered

	def _apply_noise_into_training(self, image, channel_idx: int):
		x = np.arange(-10, 10, dtype=np.float32)
		xU, xL = x + 0.5, x - 0.5
		prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL, scale=1)
		prob = prob / prob.sum()
		noise = np.random.choice(x, size=(64, 27), p=prob)
		image_threshold = np.where(image != 0, 1.0, image)
		image += noise * image_threshold * self.config['normalize_std_dev'][channel_idx]
		image = np.clip(image, 0, 10000)
		return image

	def _apply_noise_to_pressure_map(self, pressure_map):
		if self.is_train:
			pressure_map_threshold = np.where(pressure_map != 0, 1.0, pressure_map)
			pressure_map *= normalvariate(mu=1.0, sigma=self.config['add_noise'])
			scaled_std_dev = self.config['normalize_std_dev'][4] * (70.0 if self.config['normalize_per_image'] else 98.666) * self.config['add_noise']
			pressure_map += normalvariate(mu=0.0, sigma=scaled_std_dev)
			pressure_map = np.clip(pressure_map, 0, 10000)
			pressure_map *= pressure_map_threshold
			pressure_map = gaussian_filter(pressure_map, sigma=normalvariate(mu=0.5, sigma=self.config['add_noise']))
		else:
			pressure_map = gaussian_filter(pressure_map, sigma=0.5)
			pressure_map = np.clip(pressure_map, 0, 10000 if self.config['normalize_per_image'] else 100 * self.config['normalize_std_dev'][4])
		return pressure_map

	def _create_contact_map(self, pressure_map):
		return np.where(pressure_map != 0, 100.0 * self.config['normalize_std_dev'][0], pressure_map)

	def _include_estimated_depth_contact_maps(self, file_data, data_idx, input_x):
		depth_map_estimated = file_data['mdm_est'][data_idx]
		depth_map_estimated_positive = np.maximum(depth_map_estimated, 0)
		depth_map_estimated_negative = np.maximum(-depth_map_estimated, 0)
		contact_map_estimated = file_data['cm_est'][data_idx] * 100.0
		depth_map_estimated_positive *= self.config['normalize_std_dev'][1]
		depth_map_estimated_negative *= self.config['normalize_std_dev'][2]
		contact_map_estimated *= self.config['normalize_std_dev'][3]
		if self.config['use_hover']:
			depth_map_estimated_positive = np.zeros_like(depth_map_estimated_positive)
		input_x = np.concatenate((input_x, [depth_map_estimated_positive, depth_map_estimated_negative, contact_map_estimated]), axis=0)
		return input_x

	def _include_depth_contact_maps(self, file_data, data_idx, input_x):
		depth_map = file_data['mesh_depth'][data_idx].astype(np.float32)
		contact_map = file_data['mesh_contact'][data_idx].astype(np.float32)
		depth_map *= self.config['normalize_std_dev'][6]
		contact_map *= self.config['normalize_std_dev'][7]
		input_x = np.concatenate((input_x, [depth_map, contact_map]), axis=0)
		return input_x

	def _include_height_weight(self, file_data, data_idx):
		weight = file_data['body_mass'][data_idx] * self.config['normalize_std_dev'][8]
		weight_channel = np.full(shape=(1, 64, 27), fill_value=weight)
		height = (file_data['body_height'][data_idx] - 1.0) * 100 * self.config['normalize_std_dev'][9]
		height_channel = np.full(shape=(1, 64, 27), fill_value=height)
		return np.concatenate((weight_channel, height_channel), axis=0)

	def _concatenate_input_channels(self, file_data, data_idx, pm_contact_map, pressure_map, pm_sobel_filtered):
		input_x = np.zeros((0, 64, 27), dtype=np.float32)

		if not self.config['omit_contact_sobel']:
			input_x = np.concatenate((input_x, [pm_contact_map]), axis=0)
			
		if self.config['mod'] == 2:
			input_x = self._include_estimated_depth_contact_maps(file_data, data_idx, input_x)

		if self.config['omit_contact_sobel']:
			input_x = np.concatenate((input_x, [pressure_map]), axis=0)
		else:
			input_x = np.concatenate((input_x, [pressure_map, pm_sobel_filtered]), axis=0)

		if self.config['mod'] == 2:
			input_x = self._include_depth_contact_maps(file_data, data_idx, input_x)

		if self.config['include_weight_height']:
			input_x = np.concatenate((input_x, self._include_height_weight(file_data, data_idx)), axis=0)

		return input_x

	def _load_input(self, file_data, data_idx):
		pressure_map = file_data['images'][data_idx]
		pressure_map = self._process_pressure_map(pressure_map)

		pm_sobel_filtered = self._apply_sobel_filter(pressure_map)

		pressure_map *= self.config['normalize_std_dev'][4]

		if self.config['add_noise'] != 0:
			pressure_map = self._apply_noise_to_pressure_map(pressure_map)

		pm_contact_map = self._create_contact_map(pressure_map)

		if self.is_train:
			pressure_map = self._apply_noise_into_training(pressure_map, channel_idx=4)

		input_x = self._concatenate_input_channels(file_data, data_idx, pm_contact_map, pressure_map, pm_sobel_filtered)

		# Upsample input channels by a factor of 2 (64x27 -> 128x54)
		input_x = zoom(input_x, zoom=(1, 2, 2), order=1)
		return input_x

	def _load_label(self, file_data, data_idx, gender):
		g1, g2 = (1, 0) if gender == "f" else (0, 1) if gender == "m" else (None, None)
		s1 = 1	# always 1 because we always have synthetic data

		z_adj = -0.075
		z_adj_all = np.array(24	* [0, 0, z_adj * 1000])

		label_y = np.concatenate([
			file_data['markers_xyz_m'][data_idx] * 1000 + z_adj_all,									# 0:72		(72)
			file_data['body_shape'][data_idx],															# 72:82		(10)
			file_data['joint_angles'][data_idx],														# 82:154	(72)
			file_data['root_xyz_shift'][data_idx] + np.array([0, 0, z_adj]),							# 154:157	(3)
			[g1], [g2], [s1],																			# 157:160	(3)
			[file_data['body_mass'][data_idx] * self.config['normalize_std_dev'][8]],					# 160		(1)
			[(file_data['body_height'][data_idx] - 1.0) * 100 * self.config['normalize_std_dev'][9]]])	# 161		(1)

		if self.config['mod'] == 2:
			label_y = np.concatenate([
    			label_y,
				file_data['betas_est'][data_idx][:10],
				file_data['angles_est'][data_idx][:72],
				file_data['root_xyz_est'][data_idx][:3],
				file_data['root_atan2_est'][data_idx][:6]])
		return label_y

	def __getitem__(self, idx):
		file_path, data_idx, gender = self.data_indices[idx]
		file_data = load_pickle(file_path)

		# Load the input data
		input_x = self._load_input(file_data, data_idx)

		# Load the label data
		label_y = self._load_label(file_data, data_idx, gender)

		# Convert the input and label data to PyTorch tensors
		return torch.from_numpy(input_x).to(torch.float32), torch.from_numpy(label_y).to(torch.float32)

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, split='train', transform=None):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            split (str): 'train' or 'test' to load the respective dataset.
            transform (callable, optional): Optional transform to apply to inputs.
        """
        self.hdf5_file_path = hdf5_file_path
        self.split = split
        self.transform = transform
        
        # Open the file to get keys
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            self.groups = []
            self.lengths = []
            
            for key in hdf5_file[split]:  # Iterate over different body postures
                for gender in hdf5_file[f'{split}/{key}']:  # Iterate over male/female
                    inputs_path = f'{split}/{key}/{gender}/inputs'
                    labels_path = f'{split}/{key}/{gender}/labels'
                    
                    if inputs_path in hdf5_file and labels_path in hdf5_file:
                        self.groups.append((inputs_path, labels_path))
                        self.lengths.append(hdf5_file[inputs_path].shape[0])

            self.cumulative_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0)
            self.total_size = self.cumulative_lengths[-1].item()
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            # Find the right dataset based on cumulative lengths
            # dataset_idx = next(i for i, length in enumerate(self.cumulative_lengths) if idx < length)
            for i, length in enumerate(self.cumulative_lengths):
                if idx < length:
                    dataset_idx = i
                    break
            
            if dataset_idx > 0:
                idx = idx - self.cumulative_lengths[dataset_idx - 1].item()
            
            inputs_path, labels_path = self.groups[dataset_idx]
            
            input_data = hdf5_file[inputs_path][idx]
            label_data = hdf5_file[labels_path][idx]
            
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            label_tensor = torch.tensor(label_data, dtype=torch.float32)
            
            if self.transform:
                input_tensor = self.transform(input_tensor)
            
            return input_tensor, label_tensor