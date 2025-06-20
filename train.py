import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from datasets import HDF5Dataset
from models import PressureNet
from smpl_class import SMPLPreloader
import smplx
from time import time
from tqdm import tqdm
import pickle as pkl
from utils import print_error_summary, retrieve_data_file_paths, plot_input_channels, format_stats, print_mean_of_model_weights_and_gradients, get_preprocessed_hdf5_path
from datetime import datetime
import os
from torch.amp import GradScaler, autocast
from torchinfo import summary
from torch.utils.data import Subset
import random

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

def train(model, train_loader, optimizer, criterion, device, config, smpl_preloader, epoch, scaler):
	# torch.autograd.set_detect_anomaly(True)
	print(f"\nTraining ...")
	# print_mean_of_model_weights_and_gradients(model, "Before Epoch Update")

	model.train()
	running_loss = 0.0
	total_samples = 0

	# with tqdm(train_loader, desc=f"Training Epoch {epoch}/{config['num_epochs']}", unit="batch") as tepoch:
	# 	for train_batch_index, (inputs, true_labels) in enumerate(tepoch, 1):
	for train_batch_index, (inputs, true_labels) in enumerate(train_loader, 1):
		print(f"Batch: {train_batch_index:02}/{len(train_loader):02}", end='\t')

		inputs, true_labels = inputs.to(device), true_labels.to(device)
		batch_size = inputs.shape[0]	# Get the actual batch size
		total_samples += batch_size		# Keep track of the total number of samples

		optimizer.zero_grad()	# Clear gradients

		# Forward pass
		with autocast(device_type=device.type):
			predicted_labels = model(inputs)	# (B, 91)

			# Pass CNN model output through SMPL to get joint positions
			predicted_joint_positions = smpl_preloader.smpl_forward(predicted_labels, true_labels)	# (B, 24, 3)
			# # (Replace) Directly use predicted joints
			# predicted_joint_positions = predicted_labels[:, :72].view(-1, 24, 3)

			# Fetch the true joint positions from the true labels
			true_joint_positions = true_labels[:, 0:72].view(-1, 24, 3)	# (B, 24, 3)
			# # (Replace) Normalize target joints (min-max scaling or divide by a factor)
			# true_joint_positions = true_labels[:, :72] / 1000.0
			# true_joint_positions = true_joint_positions.view(-1, 24, 3)

			# # Print the stats of predicted and true joint positions (for debugging)
			# format_stats(predicted_joint_positions, "Pred Joint Positions")
			# format_stats(true_joint_positions, "True Joint Positions")

			# Calculate the losses
			joint_loss = criterion(predicted_joint_positions, true_joint_positions)

			# Add parameter loss (optional)
			param_loss = F.l1_loss(predicted_labels[:, 0:10], true_labels[:, 72:82]) \
						+ F.l1_loss(predicted_labels[:, 10:13], true_labels[:, 154:157]) \
						+ F.l1_loss(predicted_labels[:, 22:91], true_labels[:, 85:154])

			# Combine the losses
			total_loss = joint_loss + config.get('w_param', 0.1) * param_loss
			# total_loss = joint_loss

		# Backward pass and optimization (Outside the autocast context)
		scaler.scale(total_loss).backward()
		scaler.step(optimizer)
		scaler.update()

		running_loss += total_loss.item() * batch_size	# Accumulate loss

		# Logging after computation to avoid issues with mixed precision
		print(f"Joint: {joint_loss.item():.4f}", end='\t')
		print(f"Parameter: {param_loss.item():.4f}", end='\t')
		print(f"Total: {total_loss.item():.4f}", end='\t')
		print(f"Running: {running_loss / total_samples:.4f}")
	print_mean_of_model_weights_and_gradients(model, "After Epoch Update")
			# tepoch.set_postfix(loss=loss.item())
	return running_loss / total_samples	# Average loss per sample

def validate(model, valid_loader, criterion, device, config, smpl_preloader):
	print(f"\nValidating ...")

	running_loss = 0.0
	total_samples = 0

	model.eval()
	with torch.no_grad():
		# with tqdm(valid_loader, desc="Validating", unit="batch") as vepoch:
		# 	for valid_batch_index, (inputs, true_labels) in enumerate(vepoch, 1):
		for valid_batch_index, (inputs, true_labels) in enumerate(valid_loader, 1):
			print(f"Batch: {valid_batch_index:02}/{len(valid_loader):02}", end='\t')

			inputs, true_labels = inputs.to(device), true_labels.to(device)
			batch_size = inputs.shape[0]	# Get the actual batch size
			total_samples += batch_size		# Keep track of the total number of samples

			# Forward pass
			with autocast(device_type=device.type):
				predicted_labels = model(inputs)

				# Pass CNN model output through SMPL to get joint positions
				predicted_joint_positions = smpl_preloader.smpl_forward(predicted_labels, true_labels)
				# # (Replace) Directly use predicted joints
				# predicted_joint_positions = predicted_labels[:, :72].view(-1, 24, 3)

				# Fetch the true joint positions from the true labels
				true_joint_positions = true_labels[:, 0:72].view(-1, 24, 3)

				# # (Replace) Normalize target joints (min-max scaling or divide by a factor)
				# true_joint_positions = true_labels[:, :72] / 1000.0
				# true_joint_positions = true_joint_positions.view(-1, 24, 3)

				# format_stats(predicted_joint_positions, "Pred Joint Positions")
				# format_stats(true_joint_positions, "True Joint Positions")

				# Calculate the losses
				joint_loss = criterion(predicted_joint_positions, true_joint_positions)

				# Optional param loss (we usually omit this in validation)
				total_loss = joint_loss

			running_loss += total_loss.item() * batch_size	# Accumulate loss
			print(f"Joint: {joint_loss.item():.4f}", end='\t')
			print(f"Running: {running_loss / total_samples:.4f}")
	return running_loss / total_samples		# Average loss per sample

def main():
    # 0. Initializations and Configurations
	# 0.1. Parse the command line arguments
	parser = argparse.ArgumentParser(description='Train PressureNet Model')
	parser.add_argument('--mod', type=int, choices=[1, 2], required=True, help='choose a network (1 or 2)')
	parser.add_argument('--pmr', action='store_true', default=False, help='run PMR on input & precomputed spatial maps')
	parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
	parser.add_argument('--use_relu', action='store_true', default=False, help='use ReLU in place of Tanh in middle layers of CNN')
	args = parser.parse_args()

	# 0.2. Set the configuration parameters
	config = {
		'mod':					args.mod,
		'pmr':					args.pmr,
		'verbose':				args.verbose,
		'use_relu':				args.use_relu,

		# Not in args
		'batch_size':			3000,
		'num_epochs':			5,
		'learning_rate':		0.0001,	# 0.00002
		'half_betas_loss':		False,	# Halve the loss for betas
		'use_root_loss':		False,	# Use the root rotation loss
		'save_model_every':		2,
		'w_param':				0.2,	# Weight for parameter-level supervision

		# For DataLoader
		'pin_memory':			True,	# the data loader will copy Tensors into device/CUDA pinned memory before returning them.
		'num_workers_train':	28,		# how many subprocesses to use for data loading (default: 0)
		'num_workers_valid':	28,		# use 0 for validation to avoid unnecessary overhead (os.cpu_count() - 2)
		'prefetch_factor_train':2,		# no. of batches loaded in advance by each worker (default: 2 if num_workers > 0)
		'prefetch_factor_valid':2,	# no. of batches loaded in advance by each worker (default: None if num_workers == 0)
		'persistent_workers_train':	True,	# the data loader will not shut down the worker processes after a dataset has been consumed once.
		'persistent_workers_valid':	True,	# this allows to maintain the workers Dataset instances alive (default: False)
	}

	# 0.4. Check if CUDA is available
	is_cuda_available = torch.cuda.is_available()
	device = torch.device("cuda" if is_cuda_available else "cpu")

	# Print the device information
	print(f"Device (CUDA/CPU):  {device}")
	if is_cuda_available:
		print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
		print(f"Device Count:       {torch.cuda.device_count()}")
		print(f"Current Device:     {torch.cuda.current_device()}")
	else:
		print("CUDA is not available, using CPU.")

	# 0.5. Create a unique directory using timestamp for saving the best model, and snapshots of model & losses
	run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	run_dir = f"runs/run_{run_timestamp}"
	os.makedirs(run_dir, exist_ok=True)


	# 1. Data Preparation

	# Create the train and valid datasets and data loaders
	# HDF5 file paths
	# hdf5_file_name = 'preprocessed_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
	# hdf5_file_name = 'preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'
	# hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True_no_75mm.hdf5'
	hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

	hdf5_file_path = get_preprocessed_hdf5_path(hdf5_file_name)

	train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train')
	valid_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='test')

	# # Limit to small number of samples (e.g., 3000 for training, 1000 for validation)
	# # train_indices = random.sample(range(len(train_dataset)), k=min(3000, len(train_dataset)))
	# # valid_indices = random.sample(range(len(valid_dataset)), k=min(1000, len(valid_dataset)))
	# train_indices = list(range(min(3000, len(train_dataset))))
	# valid_indices = list(range(min(1000, len(valid_dataset))))

	# train_dataset = Subset(train_dataset, train_indices)
	# valid_dataset = Subset(valid_dataset, valid_indices)

	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False,	num_workers=config['num_workers_train'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor_train'], persistent_workers=config['persistent_workers_train'])
	valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False,num_workers=config['num_workers_valid'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor_valid'], persistent_workers=config['persistent_workers_valid'])

	print(f"Learning Rate:				{config['learning_rate']}")
	print(f"Mod:					{config['mod']}")
	print(f"PMR:					{config['pmr']}")
	print(f"Number of Train Examples:		{len(train_dataset)}")
	print(f"Number of Valid Examples:		{len(valid_dataset)}")
	print(f"Number of Train Batches:		{len(train_loader)}")
	print(f"Number of Valid Batches:		{len(valid_loader)}")
	print(f"Batch Size:				{config['batch_size']}")
	print(f"Number of Epochs:			{config['num_epochs']}")
	print(f"num_workers (train):			{config['num_workers_train']}")
	print(f"num_workers (valid):			{config['num_workers_valid']}")
	print(f"prefetch_factor (train):		{config['prefetch_factor_train']}")
	print(f"prefetch_factor (valid):		{config['prefetch_factor_valid']}")
	print(f"persistent_workers (train):		{config['persistent_workers_train']}")
	print(f"persistent_workers (valid):		{config['persistent_workers_valid']}")
	print(f"pin_memory (both):			{config['pin_memory']}")


	# 2. Define the model, optimizer, and loss functions
	model = PressureNet(in_channels=train_dataset[0][0].shape[0], num_classes=88, use_relu=config['use_relu']).to(device)
	optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.0005)
	criterion1 = nn.L1Loss()
	criterion2 = nn.MSELoss()

	if config['verbose']:
		print("\nModel Summary:")
		print(model)
		print()
		summary(model, input_size=(config['batch_size'], train_dataset[0][0].shape[0], 128, 54), device=device.type)

	# 3. Load SMPL models
	smpl_male_model_path = 'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
	smpl_feml_model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

	smpl_male_model = smplx.SMPL(smpl_male_model_path).to(device)
	smpl_feml_model = smplx.SMPL(smpl_feml_model_path).to(device)


	# 4. Training Loop
	best_valid_loss = float('inf')

	train_valid_losses = {
		'epoch': [],
		'train_loss': [],
		'valid_loss': [],
	}

	scaler = GradScaler(device=device.type)

	# with tqdm(range(1, config['num_epochs'] + 1), desc="Epochs", unit="epoch") as epoch_progress:
	# 	for epoch in epoch_progress:
	# 		epoch_progress.set_description(f"Epoch {epoch}/{config['num_epochs']}")
	for epoch in range(1, config['num_epochs'] + 1):
		print(f"Epoch: {epoch:03d}/{config['num_epochs']:03d}")
		print("*" * 50)

		# Initialize preloader before training loop (once per epoch)
		smpl_preloader = SMPLPreloader(smpl_male_model, smpl_feml_model, device, config)

		train_loss = train(model, train_loader, optimizer, criterion1, device, config, smpl_preloader, epoch, scaler)
		print(f"Train Loss for Epoch {epoch:03d}: {train_loss:.4f}")
		print("-" * 30)
		valid_loss = validate(model, valid_loader, criterion1, device, config, smpl_preloader)
		print(f"Valid Loss for Epoch {epoch:03d}: {valid_loss:.4f}")
		print("=" * 30)

		# Save the losses
		train_valid_losses['epoch'].append(epoch)
		train_valid_losses['train_loss'].append(train_loss)
		train_valid_losses['valid_loss'].append(valid_loss)

		# Save best model
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			best_model_path = os.path.join(run_dir, 'best_model.pth')
			torch.save(model.state_dict(), best_model_path)
			print("\nBest model saved!", end='\n\n')

		# Save the model and losses every 'save_model_every' epochs
		if epoch % config['save_model_every'] == 0 or epoch == config['num_epochs']:
			checkpoint_model_path = os.path.join(run_dir, f'checkpoint_model__epoch_{epoch}__valid_loss_{valid_loss:.4f}.pth')
			torch.save(model.state_dict(), checkpoint_model_path)
			print(f"\nCheckpoint model saved at {checkpoint_model_path}")

			checkpoint_losses_path = os.path.join(run_dir, f'checkpoint_losses__epoch_{epoch}__valid_loss_{valid_loss:.4f}.pkl')
			with open(checkpoint_losses_path, 'wb') as f:
				pkl.dump(train_valid_losses, f)
			print(f"Checkpoint losses saved at {checkpoint_losses_path}", end='\n\n')


		# if config['verbose']:
		# 	plot_input_channels(inputs_batch, train_batch_idx)

	# # 7. Load Best Model for Final Testing
	# best_model = PressureNet(in_channels=train_dataset[0][0].shape[0], num_classes=88, use_relu=config['use_relu']).to(device)
	# best_model.load_state_dict(torch.load("best_model.pth"))

if __name__ == '__main__':
	main()
