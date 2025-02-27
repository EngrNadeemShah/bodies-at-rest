import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from datasets import PressurePoseDataset
from models import PressureNet
from smpl_class import SMPLPreloader
import smplx
from time import time
from tqdm import tqdm
import pickle as pkl
from utils import print_error_summary, retrieve_data_file_paths, plot_input_channels
from datetime import datetime
import os
from torch.amp import GradScaler, autocast

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

def train(model, train_loader, optimizer, criterion, device, config, smpl_preloader, epoch, scaler):
	model.train()
	running_loss = 0.0
	total_samples = 0

	with tqdm(train_loader, desc=f"Training Epoch {epoch}/{config['num_epochs']}", unit="batch") as tepoch:
		for train_batch_index, (inputs, true_labels) in enumerate(tepoch, 1):
			inputs, true_labels = inputs.to(device), true_labels.to(device)
			batch_size = inputs.shape[0]	# Get the actual batch size
			total_samples += batch_size		# Keep track of the total number of samples

			# print(f"Train Batch Index:	{train_batch_index}")
			# print(f"Train Inputs Shape:	{inputs.shape}")
			# print(f"Train Labels Shape:	{true_labels.shape}")
			# print()

			optimizer.zero_grad()	# Clear gradients

			# Forward pass
			with autocast(device_type=device.type):
				predicted_labels = model(inputs)
				predicted_labels = smpl_preloader.forward(predicted_labels, true_labels)

				# Initialize a tensor of zeros with the same shape as predicted_labels
				zeroed_labels = torch.zeros_like(predicted_labels, device=device, requires_grad=True)

				# Calculate the losses
				loss_root_rotation = criterion(predicted_labels[:, 10:16], zeroed_labels[:, 10:16]) if config['use_root_loss'] else 0.0
				loss_eucl = criterion(predicted_labels[:, 16:40], zeroed_labels[:, 16:40])	# Euclidean loss for joint positions
				loss_betas = criterion(predicted_labels[:, :10], zeroed_labels[:, :10])		# Loss for betas with optional halving
				if config['half_betas_loss']:
					loss_betas *= 0.5

				# Combine the losses
				loss = loss_root_rotation + loss_eucl + loss_betas if config['use_root_loss'] else loss_eucl + loss_betas

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				loss *= 1000		# Apply scaling factor
				running_loss += loss.item() * batch_size	# Accumulate loss

				# Update tqdm description
				tepoch.set_postfix(loss=loss.item())
		return running_loss / total_samples	# Average loss per sample

			# if train_batch_index % 10 == 0:
			# 	# print_error_summary(true_labels[:, :72], predicted_label_markers_xyz_detached)

			# 	print(f"Train Epoch: {epoch} "
			# 		f"[{train_batch_index * config['batch_size']}/{len(train_loader.dataset)} "
			# 		f"({100.0 * train_batch_index / len(train_loader):.0f}%)]\n"
			# 		f"\tEuclidean Loss for Joint Positions:	{1000 * loss_eucl.item():.2f}\n"
			# 		f"\tBetas Loss:				{1000 * loss_betas.item():.2f}\n"
			# 		f"\tBody/Root/Pelvis Rotation Loss:		{1000 * loss_root_rotation.item() if config['use_root_loss'] else 0.0:.2f}\n"
			# 		f"\tTotal Loss:				{loss.item():.2f}\n")

def validate(model, valid_loader, criterion, device, config, smpl_preloader):
	running_loss = 0.0
	total_samples = 0

	model.eval()
	with torch.no_grad():
		with tqdm(valid_loader, desc="Validating", unit="batch") as vepoch:
			for valid_batch_index, (inputs, true_labels) in enumerate(vepoch, 1):
				inputs, true_labels = inputs.to(device), true_labels.to(device)
				batch_size = inputs.shape[0]	# Get the actual batch size
				total_samples += batch_size		# Keep track of the total number of samples

				# Forward pass
				with autocast(device_type=device.type):
					predicted_labels = model(inputs)
					predicted_labels = smpl_preloader.forward(predicted_labels, true_labels)

					# Initialize a tensor of zeros with the same shape as predicted_labels
					zeroed_labels = torch.zeros_like(predicted_labels, device=device, requires_grad=False)

					# Calculate the losses
					loss_root_rotation = criterion(predicted_labels[:, 10:16], zeroed_labels[:, 10:16]) if config['use_root_loss'] else 0.0
					loss_eucl = criterion(predicted_labels[:, 16:40], zeroed_labels[:, 16:40])	# Euclidean loss for joint positions
					loss_betas = criterion(predicted_labels[:, :10], zeroed_labels[:, :10])		# Loss for betas with optional halving
					if config['half_betas_loss']:
						loss_betas *= 0.5

					# Combine the losses
					loss = loss_root_rotation + loss_eucl + loss_betas if config['use_root_loss'] else loss_eucl + loss_betas

					loss *= 1000		# Apply scaling factor
					running_loss += loss.item() * batch_size	# Accumulate loss
			return running_loss / total_samples		# Average loss per sample

def main():
    # 0. Initializations and Configurations
	# 0.1. Parse the command line arguments
	parser = argparse.ArgumentParser(description='Train PressureNet Model')
	parser.add_argument('--add_noise', type=float, default=0, help='amount of noise to add, 0 means no noise')	# 0.1
	parser.add_argument('--include_weight_height', action='store_true', default=False, help='include height and weight as input channels')
	parser.add_argument('--omit_contact_sobel', action='store_true', default=False, help='omit contact sobel')
	parser.add_argument('--use_hover', action='store_true', default=False, help='set depth_map_estimated_positive (channel 1) to 0')
	parser.add_argument('--mod', type=int, choices=[1, 2], required=True, help='choose a network (1 or 2)')
	parser.add_argument('--pmr', action='store_true', default=False, help='run PMR on input & precomputed spatial maps')
	parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
	parser.add_argument('--use_relu', action='store_true', default=False, help='use ReLU in place of Tanh in middle layers of CNN')
	args = parser.parse_args()

	# 0.2. Set the configuration parameters
	config = {
		'add_noise':			args.add_noise,
		'include_weight_height':args.include_weight_height,
		'omit_contact_sobel':	args.omit_contact_sobel,
		'use_hover':			args.use_hover,
		'mod':					args.mod,
		'pmr':					args.pmr,
		'verbose':				args.verbose,
		'use_relu':				args.use_relu,

		# Not in args
		'batch_size':			512,
		'num_epochs':			100,
		'learning_rate':		0.00002,
		'normalize_per_image':	True,
		'half_betas_loss':		False,	# Halve the loss for betas
		'use_root_loss':		False,	# Use the root rotation loss
		'save_model_every':		10,
	}

	# 0.3. Normalization standard deviations for each channel
	if config['normalize_per_image']:
		config['normalize_std_dev'] = np.ones(10, dtype=np.float32)
	else:
		config['normalize_std_dev'] = np.array([
			1 / 41.8068,	# contact
			1 / 16.6955,	# pos est depth
			1 / 45.0851,	# neg est depth
			1 / 43.5580,	# cm est
			1 / 11.7015,	# pressure map
			1 / 45.6164 if config['add_noise'] else 1 / 29.8036,	# pressure map sobel
			1,  # OUTPUT DO NOTHING
			1,  # OUTPUT DO NOTHING
			1 / 30.2166,  # weight
			1 / 14.6293   # height
		])

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

	# Set the number of workers and pin memory based on the device
	config['num_workers'] = 4 if device.type == 'cuda' else os.cpu_count()
	config['pin_memory'] = device.type == 'cuda'

	# 0.5. Create the directories for saving the model and losses
	os.makedirs('checkpoints', exist_ok=True)
	os.makedirs('losses', exist_ok=True)


	# 1. Data Preparation
	# Define the path to the original train and test data (.pickle files)
	if config['mod'] == 1:
		train_files_dir = 'synthetic_data/original/train'
		valid_files_dir = 'synthetic_data/original/test'
	elif config['mod'] == 2:
		train_files_dir = 'synthetic_data/by_mod1/a/train'
		valid_files_dir = 'synthetic_data/by_mod1/a/test'

	# Get the paths to the train and test data files
	train_file_paths = retrieve_data_file_paths(train_files_dir)
	valid_file_paths = retrieve_data_file_paths(valid_files_dir)

	# Create the train and test datasets and data loaders
	train_dataset = PressurePoseDataset(file_paths=train_file_paths, config=config, is_train=True)
	valid_dataset = PressurePoseDataset(file_paths=valid_file_paths, config=config, is_train=False)
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
	valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

	print(f"Number of Train Examples:		{len(train_dataset)}")
	print(f"Number of Valid Examples:		{len(valid_dataset)}")
	print(f"Number of Train Batches:		{len(train_loader)}")
	print(f"Number of Valid Batches:		{len(valid_loader)}")
	print(f"Batch Size:				{config['batch_size']}")
	print(f"Number of Epochs:			{config['num_epochs']}")


	# 2. Define the model, optimizer, and loss functions
	model = PressureNet(in_channels=train_dataset[0][0].shape[0], num_classes=88, use_relu=config['use_relu']).to(device)
	optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.0005)
	criterion1 = nn.L1Loss()
	criterion2 = nn.MSELoss()


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

	with tqdm(range(1, config['num_epochs'] + 1), desc="Epochs", unit="epoch") as epoch_progress:
		for epoch in epoch_progress:
			epoch_progress.set_description(f"Epoch {epoch}/{config['num_epochs']}")

			# Initialize preloader before training loop (once per epoch)
			smpl_preloader = SMPLPreloader(smpl_male_model, smpl_feml_model, device, config)

			train_loss = train(model, train_loader, optimizer, criterion1, device, config, smpl_preloader, epoch, scaler)
			valid_loss = validate(model, valid_loader, criterion1, device, config, smpl_preloader)

			# Save the losses
			train_valid_losses['epoch'].append(epoch)
			train_valid_losses['train_loss'].append(train_loss)
			train_valid_losses['valid_loss'].append(valid_loss)

			# Save best model
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(), 'checkpoints/best_model.pth')
				print("Best model saved!")

			# Save the model and losses every 'save_model_every' epochs
			if epoch % config['save_model_every'] == 0 or epoch == config['num_epochs']:
				timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
				model_path = f'checkpoints/model_epoch_{epoch}_valid_loss_{valid_loss:.4f}_{timestamp}.pth'
				torch.save(model.state_dict(), model_path)
				print(f"Model saved at {model_path}")

				losses_path = f"losses/losses_epoch_{epoch}_{timestamp}.p"
				with open(losses_path, 'wb') as f:
					pkl.dump(train_valid_losses, f)
				print(f"Losses saved at {losses_path}")


			# if config['verbose']:
			# 	plot_input_channels(inputs_batch, train_batch_idx)

	# # 7. Load Best Model for Final Testing
	# best_model = PressureNet(in_channels=train_dataset[0][0].shape[0], num_classes=88, use_relu=config['use_relu']).to(device)
	# best_model.load_state_dict(torch.load("best_model.pth"))

if __name__ == '__main__':
	main()
