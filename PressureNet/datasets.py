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
mpl.rcParams['text.usetex'] = False  # Disable LaTeX rendering
mpl.rcParams['font.family'] = 'DejaVu Sans'  # Set default font family
mpl.use('TkAgg')

def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f, encoding='latin1')


class DatasetClass(Dataset):
	def __init__(self, file_list, creation_type, test, verbose=False, config=None, mat_size=(64, 27), blur_sigma=0.5):
		"""
		Args:
			file_list (list of str): List of file paths to load data from.
			creation_type (str): Type of data to load, e.g., 'synth' or 'real'.
			test (bool): Whether the dataset is for testing.
			verbose (bool): If True, print dataset statistics after initialization.

			CTRL_PNL (dict): Dictionary of control panel parameters.
			mat_size (tuple): Size of the pressure mat.
			blur_sigma (float): Standard deviation of the Gaussian blur.
			z_adj (float): Adjustment to the z-coordinate of the markers.
		"""
		self.file_list = [f for f in file_list if creation_type in f]
		self.creation_type = creation_type
		self.test = test
		self.verbose = verbose
		self.config = config
		self.mat_size = mat_size
		self.blur_sigma = blur_sigma
		self.data_indices = self._index_data()
		self.z_adj = -0.075 if self.creation_type == 'synth' else 0.0 if self.creation_type == 'real' else None
		print()

	def _index_data(self):
		"""
		Index the data across all files without loading it into memory.
		Returns a list of tuples (file_path, key, index) for efficient lookup.
		"""
		data_indices = []
		for f_idx, file_path in enumerate(self.file_list):

			# Identify the gender from the file path
			gender = 'f' if '_f_' in file_path else 'm' if '_m_' in file_path else None

			# Load the pickle file
			file_data = load_pickle(file_path)

			if self.verbose:
				pass
				# print(f'File {f_idx + 1:02d}/{len(self.file_list):02d}: {file_path}')
				# print(f'No. of keys: {len(file_data.keys())} -> {file_data.keys()}')
				# print(f'No. of images: {len(file_data["images"])}')

			# Index the data based on the key 'images'
			for data_idx in range(len(file_data['images'])):

				# Skip the second half of the data for testing
				if self.test:
					if len(file_data['images']) == 3000 and data_idx >= len(file_data['images']) / 2:
						pass
					elif len(file_data['images']) == 1500 and data_idx >= len(file_data['images']) / 3:
						pass
					else:
						data_indices.append((file_path, data_idx, gender))

				else:
					data_indices.append((file_path, data_idx, gender))

		return data_indices

	def __len__(self):
		return len(self.data_indices)

	def _load_input(self, file_data, data_idx):
		"""
		Load and preprocess input data for the model.

		This method performs several preprocessing steps on the input data, including:
		- Plotting input images and output labels if verbose mode is enabled.
		- Clipping, reshaping, applying Gaussian blur, and normalizing the pressure map.
		- Applying Sobel filter to the pressure map and concatenating it with the original pressure map.
		- Optionally concatenating estimated depth map and contact map with the input.
		- Creating a contact map from the pressure map and stacking it with the input.
		- Optionally concatenating depth map and contact map labels with the input.
		- Normalizing the input using standard deviation coefficients.

		Args:
			file_data (dict): Dictionary containing the input data.
			data_idx (int): Index of the data sample to load.

		Returns:
			np.ndarray: Preprocessed input data ready for the model.
		"""

		# if self.verbose:
		# 	if self.config['adjust_ang_from_est']:
		# 		input_keys	= ['images', 'mesh_contact', 'mesh_depth', 'cm_est', 'mdm_est']
		# 		output_keys	= ['markers_xyz_m', 'body_shape', 'joint_angles', 'root_xyz_shift', 'body_mass', 'body_height', 'betas_est', 'angles_est', 'root_xyz_est', 'root_atan2_est']
		# 	else:
		# 		input_keys	= ['images', 'mesh_contact', 'mesh_depth']
		# 		output_keys	= ['markers_xyz_m', 'body_shape', 'joint_angles', 'root_xyz_shift', 'body_mass', 'body_height']

		# 	# Plot the input images
		# 	fig, axes = plt.subplots(1, len(input_keys), figsize=(15, 5))
		# 	for i, key in enumerate(input_keys):
		# 		title = f"{key} - {file_data[key][data_idx].shape}"
		# 		pm_image = file_data[key][data_idx].reshape(self.mat_size) if key == 'images' else file_data[key][data_idx]
		# 		axes[i].imshow(pm_image)
		# 		axes[i].set_title(title)
		# 	plt.show()


		# 	# Plot the output labels (markers_xyz_m) on top of the input images
		# 	fig, axes = plt.subplots(1, len(input_keys), figsize=(15, 5))
		# 	for i, key in enumerate(input_keys):
		# 		pm_image = file_data[key][data_idx].reshape(self.mat_size) if key == 'images' else file_data[key][data_idx]
		# 		markers = file_data['markers_xyz_m'][data_idx].reshape(-1, 3)
		# 		markers_x, markers_y = markers[:, 0], markers[:, 1]
		# 		markers_x = (markers_x - markers_x.min()) / (markers_x.max() - markers_x.min())
		# 		markers_y = (markers_y - markers_y.min()) / (markers_y.max() - markers_y.min())
		# 		markers_x = markers_x * self.mat_size[1]
		# 		markers_y = markers_y * self.mat_size[0]
		# 		markers_y = self.mat_size[0] - markers_y
		# 		axes[i].imshow(pm_image)
		# 		axes[i].scatter(markers_x, markers_y, c='r', s=5)
		# 		axes[i].set_title(f'{key} with Markers')
		# 	plt.show()

		# Create a dictionary to store different input channels for visualization
		visualization_dict = {}

		# Generate noise for the pressure map and pressure map Sobel filtered (only for training)
		if not self.test:
			x = np.arange(-10, 10, dtype=np.float32)
			xU, xL = x + 0.5, x - 0.5
			prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL, scale=1)	# Scale is the standard deviation using a cumulative density function
			prob = prob / prob.sum()	# Normalize the probabilities so their sum is 1
			noise = np.random.choice(x, size=self.mat_size, p=prob)

		# Load the pressure map and clip, reshape, apply Gaussian blur, and normalize it
		pressure_map = file_data['images'][data_idx] * self.config['pmat_mult']
		pressure_map = np.clip(pressure_map, 0, 100).reshape(self.mat_size)
		visualization_dict['PM Initial'] = np.copy(pressure_map)

		if not self.config['cal_noise']:
			pressure_map = gaussian_filter(pressure_map, sigma=self.blur_sigma)
			visualization_dict['PM Gaussian'] = np.copy(pressure_map)

		pressure_map = np.clip(pressure_map, 0, 100)

		if self.config['normalize_per_image']:
			pressure_map = pressure_map * 20000.0 / pressure_map.sum()
			visualization_dict['PM Normalized'] = np.copy(pressure_map)

		if self.mat_size == (84, 47):
			pressure_map = pressure_map[10:74, 10:37]

		# Apply Sobel filter to the pressure map
		sx = sobel(pressure_map, axis=0, mode='constant')
		sy = sobel(pressure_map, axis=1, mode='constant')
		pressure_map_sobel_filtered = np.hypot(sx, sy)
		visualization_dict['PM Sobel Initial'] = np.copy(pressure_map_sobel_filtered)


		# Normalize the pressure map using standard deviation coefficients
		pressure_map *= self.config['norm_std_coeffs'][4]


		if self.config['cal_noise']:
			if not self.test:
				pressure_map_binary = np.where(pressure_map != 0, 1.0, pressure_map)
				pressure_map *= normalvariate(mu=1.0, sigma=self.config['cal_noise_amt'])
				noise_sigma = self.config['norm_std_coeffs'][4] * (70.0 if self.config['normalize_per_image'] else 98.666) * self.config['cal_noise_amt']
				pressure_map += normalvariate(mu=0.0, sigma=noise_sigma)
				pressure_map = np.clip(pressure_map, 0, 10000)
				pressure_map *= pressure_map_binary
				pressure_map = gaussian_filter(pressure_map, sigma=normalvariate(mu=0.5, sigma=self.config['cal_noise_amt']))

			else:
				pressure_map = gaussian_filter(pressure_map, sigma=0.5)
				pressure_map = np.clip(pressure_map, 0, 10000 if self.config['normalize_per_image'] else 100 * self.config['norm_std_coeffs'][4])


		# Create contact map from the pressure map and stack in the input as the channel 0
		if self.config['cal_noise']:
			pressure_map_contact_mask = np.where(pressure_map != 0, 100.0 * self.config['norm_std_coeffs'][0], pressure_map)

		else:	# (old flag: config['incl_pmat_cntct_input'])
			pressure_map_contact_mask = (pressure_map > 0) * 100.0
			pressure_map_contact_mask *= self.config['norm_std_coeffs'][0]

		visualization_dict['PM Contact Mask x100'] = np.copy(pressure_map_contact_mask)


		# Add noise to the pressure_map
		if not self.test:
			pressure_map_mask_binary = np.where(pressure_map != 0, 1.0, pressure_map)
			pressure_map += noise * pressure_map_mask_binary * self.config['norm_std_coeffs'][4]
			pressure_map = np.clip(pressure_map, 0, 10000)	# Clip the pressure map to the sensor limits


		# Normalize the pressure map Sobel filtered using standard deviation coefficients
		if self.config['clip_sobel']:
			pressure_map_sobel_filtered = np.clip(pressure_map_sobel_filtered, 0, 100)

		if self.config['normalize_per_image']:
			pressure_map_sobel_filtered *= 20000.0 / pressure_map_sobel_filtered.sum()
			visualization_dict['PM Sobel Normalized'] = np.copy(pressure_map_sobel_filtered)

		if self.config['cal_noise']:
			pressure_map_sobel_filtered *= normalvariate(mu=1.0, sigma=self.config['cal_noise_amt'])

		pressure_map_sobel_filtered *= self.config['norm_std_coeffs'][5]


		# Add noise to the pressure_map_sobel_filtered
		if not self.test:
			pressure_map_sobel_filtered_mask_binary = np.where(pressure_map_sobel_filtered != 0, 1.0, pressure_map_sobel_filtered)
			pressure_map_sobel_filtered += noise * pressure_map_sobel_filtered_mask_binary * self.config['norm_std_coeffs'][5]


		# Concatenate the 3-channel estimated depth map and contact map with the 2-channel input
		if self.config['depth_map_input_est']:
			depth_map_estimated = file_data['mdm_est'][data_idx]
			depth_map_estimated_positive = np.maximum(depth_map_estimated, 0)
			depth_map_estimated_negative = np.maximum(-depth_map_estimated, 0)
			contact_map_estimated_x100 = file_data['cm_est'][data_idx] * 100.0

			depth_map_estimated_positive *= self.config['norm_std_coeffs'][1]
			depth_map_estimated_negative *= self.config['norm_std_coeffs'][2]
			contact_map_estimated_x100 *= self.config['norm_std_coeffs'][3]

			visualization_dict['DepthMap Est']			= np.copy(depth_map_estimated)
			visualization_dict['DepthMap Est Pos']		= np.copy(depth_map_estimated_positive)
			visualization_dict['DepthMap Est Neg']		= np.copy(depth_map_estimated_negative)
			visualization_dict['ContactMap Est x100']	= np.copy(contact_map_estimated_x100)


		# Concatenate the 2-channel depth map and contact map with the input
		if (self.test and self.config['depth_map_labels_test']) or (not self.test and self.config['depth_map_labels']):
			depth_map = file_data['mesh_depth'][data_idx].astype(np.float32)
			contact_map = file_data['mesh_contact'][data_idx].astype(np.float32)

			depth_map *= self.config['norm_std_coeffs'][6]
			contact_map *= self.config['norm_std_coeffs'][7]

			visualization_dict['DepthMap'] = np.copy(depth_map)
			visualization_dict['ContactMap'] = np.copy(contact_map)


		# Concatenate only the necessary input channels (if they exist)
		# The order should be:
		# mod 2 | mod 1
			# 0 | 0: pressure_map_contact_mask,
			# 1 | _: depth_map_estimated_positive,
			# 2 | _: depth_map_estimated_negative,
			# 3 | _: contact_map_estimated_x100,
			# 4 | 1: pressure_map,
			# 5 | 2: pressure_map_sobel_filtered,
			# 6 | _: depth_map,
			# 7 | _: contact_map.

		input_x = np.array([pressure_map_contact_mask])

		if self.config['depth_map_input_est']:
			input_x = np.concatenate((input_x, [depth_map_estimated_positive, depth_map_estimated_negative, contact_map_estimated_x100]), axis=0)

		input_x = np.concatenate((input_x, [pressure_map, pressure_map_sobel_filtered]), axis=0)

		if (self.test and self.config['depth_map_labels_test']) or (not self.test and self.config['depth_map_labels']):
			input_x = np.concatenate((input_x, [depth_map, contact_map]), axis=0)


		if self.verbose:
			# Plot the visualization dictionary (contains all steps of the input processing)
			num_images = len(visualization_dict)
			num_cols = min(num_images, 5)
			num_rows = (num_images + num_cols - 1) // num_cols
			fig, axes = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))
			axes = axes.flatten() if num_rows > 1 else axes

			for i, (title, image) in enumerate(visualization_dict.items()):
				axes[i].imshow(image)
				axes[i].set_title(f'{title}\n{image.shape} | {image.dtype}\nmin: {image.min():.2f} | max: {image.max():.2f}\nmean: {image.mean():.2f} | sum: {image.sum():.2f}', fontsize=8, pad=10)
				axes[i].axis('off')

			for j in range(i + 1, len(axes)):
				axes[j].axis('off')

			plt.tight_layout()
			plt.show()
		return input_x

	def _load_label(self, file_data, data_idx, gender):
		full_body_rot = False if self.creation_type == 'real' else self.config['full_body_rot']
		g1, g2 = (1, 0) if gender == "f" else (0, 1) if gender == "m" else (None, None)
		s1 = 1 if self.creation_type == 'synth' else 0 if self.creation_type == 'real' else None

		z_adj_all = np.array(24	* [0.0, 0.0, self.z_adj * 1000])
		z_adj_one = np.array(1	* [0.0, 0.0, self.z_adj * 1000])

		if self.creation_type == 'synth':
			if self.config['loss_type'] != 'direct':

				label_y = np.concatenate([
					file_data['markers_xyz_m'][data_idx][:72] * 1000 + z_adj_all,
					file_data['body_shape'][data_idx][:10],
					file_data['joint_angles'][data_idx][:72],
					file_data['root_xyz_shift'][data_idx][:3] + np.array([0.0, 0.0, self.z_adj]),
					[g1], [g2], [s1],
					[file_data['body_mass'][data_idx]],
					[(file_data['body_height'][data_idx] - 1.) * 100]
				])

				if self.config['adjust_ang_from_est']:
					label_y = np.concatenate([
						label_y,
						file_data['betas_est'][data_idx][:10],
						file_data['angles_est'][data_idx][:72],
						file_data['root_xyz_est'][data_idx][:3]
					])

					if full_body_rot:
						label_y = np.concatenate([label_y, file_data['root_atan2_est'][data_idx][:6]])

			elif self.config['loss_type'] == 'direct':
				label_y = np.concatenate([
					np.zeros(9),
					file_data['markers_xyz_m_offset'][data_idx][3:6] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][21:24] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][18:21] * 1000 + z_adj_one,
					np.zeros(3),
					file_data['markers_xyz_m_offset'][data_idx][27:30] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][24:27] * 1000 + z_adj_one,
					np.zeros(18),
					file_data['markers_xyz_m_offset'][data_idx][:3] * 1000 + z_adj_one,
					np.zeros(6),
					file_data['markers_xyz_m_offset'][data_idx][9:12] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][6:9] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][15:18] * 1000 + z_adj_one,
					file_data['markers_xyz_m_offset'][data_idx][12:15] * 1000 + z_adj_one,
					np.zeros(6),
					np.zeros(85),
					[g1], [g2], [s1],
					[file_data['body_mass'][data_idx]],
					[(file_data['body_height'][data_idx] - 1.) * 100]
				])
				if self.config['adjust_ang_from_est']:
					label_y = np.concatenate([
						label_y,
						file_data['betas_est'][data_idx][:10],
						file_data['angles_est'][data_idx][:72],
						file_data['root_xyz_est'][data_idx][:3]
					])

		elif self.creation_type == 'real':
			label_y = np.concatenate([
				np.zeros(9),
				file_data['markers_xyz_m'][data_idx][3:6] * 1000,
				file_data['markers_xyz_m'][data_idx][21:24] * 1000,
				file_data['markers_xyz_m'][data_idx][18:21] * 1000,
				np.zeros(3),
				file_data['markers_xyz_m'][data_idx][27:30] * 1000,
				file_data['markers_xyz_m'][data_idx][24:27] * 1000,
				np.zeros(18),
				file_data['markers_xyz_m'][data_idx][:3] * 1000,
				np.zeros(6),
				file_data['markers_xyz_m'][data_idx][9:12] * 1000,
				file_data['markers_xyz_m'][data_idx][6:9] * 1000,
				file_data['markers_xyz_m'][data_idx][15:18] * 1000,
				file_data['markers_xyz_m'][data_idx][12:15] * 1000,
				np.zeros(6),
				np.zeros(85),
				[g1], [g2], [s1],
				[file_data['body_mass'][data_idx]],
				[(file_data['body_height'][data_idx] - 1.) * 100]
			])
			if self.config['adjust_ang_from_est']:
				label_y = np.concatenate([
					label_y,
					file_data['betas_est'][data_idx][:10],
					file_data['angles_est'][data_idx][:72],
					file_data['root_xyz_est'][data_idx][:3]
				])

		label_y[160] *= self.config['norm_std_coeffs'][8]
		label_y[161] *= self.config['norm_std_coeffs'][9]

		return label_y

	def __getitem__(self, idx):
		file_path, data_idx, gender = self.data_indices[idx]
		file_data = load_pickle(file_path)

		input_x = self._load_input(file_data, data_idx)

		# Upsample each channel of the input by a factor of 2, changing dimensions from (num_channels x 64 x 27) to (num_channels x 128 x 54)
		input_x = zoom(input_x, zoom=(1, 2, 2), order=1)

		label_y = self._load_label(file_data, data_idx, gender)

		# Add weight and height from the label to the input
		if self.config['incl_ht_wt_channels']:
			weight_channel = np.full(shape=(1, input_x.shape[1], input_x.shape[2]), fill_value=label_y[160], dtype=np.float32)
			height_channel = np.full(shape=(1, input_x.shape[1], input_x.shape[2]), fill_value=label_y[161], dtype=np.float32)
			input_x = np.concatenate((input_x, weight_channel, height_channel), axis=0)

		# Omit the Contact (channel 0) & Sobel (channel 2 or 5) channels from the input by setting every pixel in these channels to 0
		if self.config['omit_cntct_sobel']:
			input_x[0] = 0

			sobel_channel_index = 5 if self.config['depth_map_input_est'] else 2
			input_x[sobel_channel_index] = 0

		# Omit the depth_map_estimated_positive (channel 1) from the input by setting every pixel in this channel to 0
		if self.config['use_hover'] == False and self.config['adjust_ang_from_est'] == True:
			input_x[1] *= 0

		# Convert to PyTorch tensors
		input_x_tensor = torch.Tensor(input_x)
		label_y_tensor = torch.Tensor(label_y)

		if self.verbose:
			# Plot the final input channels
			num_channels = len(input_x)
			num_cols = min(num_channels, 5)
			num_rows = (num_channels + num_cols - 1) // num_cols
			fig, axes = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))
			axes = axes.flatten() if num_rows > 1 else axes

			for i in range(num_channels):
				axes[i].imshow(input_x[i])
				axes[i].set_title(f'Input Channel {i}\n{input_x[i].shape} | {input_x[i].dtype}\nmin: {input_x[i].min():.2f} | max: {input_x[i].max():.2f}\nmean: {input_x[i].mean():.2f} | sum: {input_x[i].sum():.2f}', fontsize=8, pad=10)
				axes[i].axis('off')

			for j in range(i + 1, len(axes)):
				axes[j].axis('off')

			plt.tight_layout()
			plt.show()

		return input_x_tensor, label_y_tensor