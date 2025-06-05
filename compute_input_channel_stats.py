import torch
from torch.utils.data import DataLoader
from datasets import HDF5Dataset
import numpy as np

def compute_channel_stats(train_loader):
	total_sum = 0.0
	total_sq_sum = 0.0
	total_pixels = 0
	num_channels = None

	for batch_index, (inputs, _) in enumerate(train_loader, 1):
		print(f"Processing Batch: {batch_index}/{len(train_loader)}")	# , end='\r'
		inputs = inputs.float()  # shape: (B, C, 128, 54)
		if num_channels is None:
			num_channels = inputs.shape[1]
			total_sum = torch.zeros(num_channels)
			total_sq_sum = torch.zeros(num_channels)

		B, C, H, W = inputs.shape
		pixels_per_channel = B * H * W

		# Sum and squared sum per channel
		sum_per_channel = inputs.sum(dim=[0, 2, 3])  # shape: (C,)
		sq_sum_per_channel = (inputs ** 2).sum(dim=[0, 2, 3])

		total_sum += sum_per_channel
		total_sq_sum += sq_sum_per_channel
		total_pixels += pixels_per_channel

	mean = total_sum / total_pixels
	std = ((total_sq_sum / total_pixels) - (mean ** 2)).sqrt()

	print("\n✅ Channel-wise mean and std:")
	for c in range(num_channels):
		print(f"[{c:02d}] mean = {mean[c]:.4f}, std = {std[c]:.4f}")

	return mean, std


if __name__ == "__main__":
	hdf5_file_path = '../../scratch/data/pre_processed/preprocessed_mod1_float32_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

	# Set the configuration parameters
	config = {
		'batch_size':			1024,

		# For DataLoader
		'pin_memory':			True,
		'num_workers_train':	4,
		'prefetch_factor_train':2,
		'persistent_workers_train':	True,
	}

	train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train')
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers_train'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor_train'], persistent_workers=config['persistent_workers_train'])
	compute_channel_stats(train_loader)
