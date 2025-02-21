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
from utils import convert_axis_angle_to_rotation_matrix, apply_global_rigid_transformations, print_error_summary, retrieve_data_file_paths, plot_input_channels

def train(model, train_loader, optimizer, criterion, device, config, epoch):
	# Define the parent array with -1 to indicate a non-existent parent (replaced 4294967295)
	parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

	vertices = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

	# When loss_type == 'anglesDC'
	bounds = torch.Tensor([
		[-0.5933865286111969, 0.5933865286111969],
		[-2*np.pi, 2*np.pi],
		[-1.215762200416361, 1.215762200416361],
		[-1.5793940868065197, 0.3097956806],
		[-0.5881754611, 0.5689768556],
		[-0.5323249722, 0.6736965222],
		[-1.5793940868065197, 0.3097956806],
		[-0.5689768556, 0.5881754611],
		[-0.6736965222, 0.5323249722],
		[-np.pi / 3, np.pi / 3],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 36, np.pi / 36],
		[-0.02268926111, 2.441713561],
		[-0.01, 0.01],
		[-0.01, 0.01],    # knee
		[-0.02268926111, 2.441713561],
		[-0.01, 0.01], [-0.01, 0.01],
		[-np.pi / 3, np.pi / 3],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		# ankle, pi/36 or 5 deg
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		# ankle, pi/36 or 5 deg
		[-np.pi / 3, np.pi / 3],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 36, np.pi / 36],
		[-0.01, 0.01],
		[-0.01, 0.01],
		[-0.01, 0.01],    # foot
		[-0.01, 0.01],
		[-0.01, 0.01],
		[-0.01, 0.01],    # foot
		[-np.pi / 3, np.pi / 3],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 36, np.pi / 36],    # neck
		[-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],
		[-2.455676183 * 1 / 3, 0.7627082389 * 1 / 3],
		[-1.570795 * 1 / 3, 2.188641033 * 1 / 3],
		[-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],
		[-0.7627082389 * 1 / 3, 2.455676183 * 1 / 3],
		[-2.188641033 * 1 / 3, 1.570795 * 1 / 3],
		[-np.pi / 3, np.pi / 3],
		[-np.pi / 36, np.pi / 36],
		[-np.pi / 36, np.pi / 36],    # head
		[-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],
		[-2.455676183 * 2 / 3, 0.7627082389 * 2 / 3],
		[-1.570795 * 2 / 3, 2.188641033 * 2 / 3],
		[-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],
		[-0.7627082389 * 2 / 3, 2.455676183 * 2 / 3],
		[-2.188641033 * 2 / 3, 1.570795 * 2 / 3],
		[-0.01, 0.01],
		[-2.570867817, 0.04799651389],
		[-0.01, 0.01],    # elbow
		[-0.01, 0.01],
		[-0.04799651389, 2.570867817],
		[-0.01, 0.01],    # elbow
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		# wrist, pi/36 or 5 deg
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		[-np.pi / 6, np.pi / 6],
		# wrist, pi/36 or 5 deg
		[-0.01, 0.01], [-0.01, 0.01],
		[-0.01, 0.01],    # hand
		[-0.01, 0.01],
		[-0.01, 0.01],
		[-0.01, 0.01]
	]) * 1.2

	# Load SMPL models
	smpl_male_model_path = 'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
	smpl_feml_model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

	smpl_male_model = smplx.SMPL(smpl_male_model_path).to(device)
	smpl_feml_model = smplx.SMPL(smpl_feml_model_path).to(device)

	# Initialize preloader before training loop (once per epoch)
	smpl_preloader = SMPLPreloader(smpl_male_model, smpl_feml_model, device, config, vertices)

	model.train()
	for train_batch_index, (inputs, true_labels) in enumerate(train_loader, 1):
		inputs, true_labels = inputs.to(device), true_labels.to(device)
		batch_size = inputs.shape[0]

		print(f"Train Batch Index:	{train_batch_index}")
		print(f"Train Inputs Shape:	{inputs.shape}")
		print(f"Train Labels Shape:	{true_labels.shape}")
		print()

		optimizer.zero_grad()
		predicted_labels = model(inputs)
		predicted_labels *= 0.01

		predicted_labels[:, 10:13] += torch.tensor([0.6, 1.2, 0.1], device=device)

		predicted_labels = F.pad(predicted_labels, (0, 3))
		predicted_labels[:, 22:91] = predicted_labels[:, 19:88].clone()
		predicted_labels[:, 19:22] = torch.atan2(predicted_labels[:, [16, 17, 18]], predicted_labels[:, [13, 14, 15]])	# pitch, yaw, roll

		predicted_labels[:, 0:10] = torch.tanh(predicted_labels[:, 0:10] / 3) * 3

		# Store the predicted labels
		predicted_label_betas = predicted_labels[:, 0:10].clone()
		predicted_label_root_xyz = predicted_labels[:, 10:13].clone()

		# Normalize for tanh activation function
		bounds_mean = bounds.mean(dim=1)		# (72, 2) -> (72,)
		bounds_diff = torch.abs(bounds[:, 0] - bounds[:, 1])
		scaled_labels = (predicted_labels[:, 19:91] - bounds_mean) * (2.0 / bounds_diff)
		tanh_labels = torch.tanh(scaled_labels)
		predicted_labels[:, 19:91] = tanh_labels / (2.0 / bounds_diff) + bounds_mean

		predicted_label_angles_rot_mat = convert_axis_angle_to_rotation_matrix(predicted_labels[:, 19:91].view(-1, 24, 3))
############################################################################################################################################################################
		genders = true_labels[:, 157:159].clone()
		shapedirs, v_template, J_regressor, posedirs, weights, BRD_shape = smpl_preloader.get_parameters(genders[:, 1])

		# SMPL Forward Pass
		# Shapes are:
		# shapedirs:	(B, 10, 6890*3)
		# v_template:	(B, 6890, 3)
		# J_regressor:	(B, 6890, 24)
		# posedirs:		(B, 207, 10*3)
		# weights:		(B, 10, 24)
		# B=10, R=6890, D=3

		# 1. Compute the shaped vertices
		# by multiplying the shape parameters with the shape directions and adding the template vertices.
		# Shape: (B, 10) -> (B, 1, 10) x (B, 10, 6890*3) -> (B, 1, 6890*3) -> (B, 6890, 3) + (B, 6890, 3) -> (B, 6890, 3)
		SMPL_pred_v_shaped = torch.bmm(predicted_label_betas.unsqueeze(1), shapedirs).squeeze(1).view(batch_size, BRD_shape[1], BRD_shape[2]) + v_template
		# Extract the vertices of interest
		# Shape: (B, 6890, 3) -> (B, 10, 3)
		SMPL_pred_v_shaped_red = torch.stack([SMPL_pred_v_shaped[:, vertex, :] for vertex in vertices], dim=1)

		# 2. Compute the posed vertices
		# Compute the pose feature
		# by subtracting the identity matrix from the rotation matrix
		# Shape: (B, 24, 3, 3) -> (B, 23, 3, 3) - (3, 3) -> (B, 23, 3, 3) -> (B, 207)
		predicted_label_angles_rot_mat_pose_feature = (predicted_label_angles_rot_mat[:, 1:] - torch.eye(3, device=device)).view(-1, 207)
		# Compute the posed vertices
		# by multiplying the pose feature with the pose directions
		# Shape: (B, 207) -> (B, 1, 207) x (B, 207, 30) -> (B, 1, 30) -> (B, 10, 3)
		SMPL_pred_v_posed = torch.bmm(predicted_label_angles_rot_mat_pose_feature.unsqueeze(1), posedirs).view(-1, 10, BRD_shape[2])
		# Add the shaped vertices to get the final vertices
		# Shape: (B, 10, 3) + (B, 10, 3) -> (B, 10, 3)
		SMPL_pred_v_posed = SMPL_pred_v_posed.clone() + SMPL_pred_v_shaped_red

		# 3. Compute the markers (joint locations) in 3D
		# Compute the joint locations
		# by multiplying the shaped vertices with the joint regressor
		# Shape: (B, 6890, 3) -> (B, 3, 6890) x (B, 6890, 24) -> (B, 3, 24) -> (B, 24, 3)
		SMPL_pred_J = torch.matmul(SMPL_pred_v_shaped.transpose(1, 2), J_regressor).permute(0, 2, 1)
		# Apply global rigid transformations to the joint locations
		# Shape: (B, 24, 3), (B, 24, 4, 4) = apply_global_rigid_transformations((B, 24, 3, 3), (B, 24, 3), (24,))
		predicted_label_markers_xyz, SMPL_pred_A = apply_global_rigid_transformations(predicted_label_angles_rot_mat, SMPL_pred_J, parents, device, rotate_base=False)
		# Compute the markers in 3D
		# by subtracting the root joint location and adding the root location
		# Shape: (B, 24, 3) - (B, 1, 3) + (B, 1, 3) -> (B, 24, 3) (Broadcasting is used to expand the dim 1 to 24)
		predicted_label_markers_xyz = predicted_label_markers_xyz - SMPL_pred_J[:, 0:1, :] + predicted_label_root_xyz.unsqueeze(1)

		# 4. Compute final vertices
		# Compute the transformation matrices:
		# Shape: (B, 10, 24) x [(B, 24, 4, 4) -> (B, 24, 16)] -> (B, 10, 16) -> (B, 10, 4, 4)
		SMPL_pred_T = torch.bmm(weights, SMPL_pred_A.view(batch_size, 24, 16)).view(batch_size, -1, 4, 4)
		# Add a homogeneous coordinate to the posed vertices
		# Shape: cat[(B, 10, 3), (B, 10, 1)] -> (B, 10, 4)
		SMPL_pred_v_posed_homo = torch.cat([SMPL_pred_v_posed, torch.ones(batch_size, SMPL_pred_v_posed.shape[1], 1)], dim=2)
		# Compute final vertices
		# Shape: (B, 10, 4, 4) x (B, 10, 4, 1) -> (B, 10, 4, 1)
		SMPL_pred_v_homo = torch.matmul(SMPL_pred_T, torch.unsqueeze(SMPL_pred_v_posed_homo, -1))
		# Shape: (B, 10, 4, 1) -> (B, 10, 3) - (B, 1, 3) + (B, 1, 3) -> (B, 10, 3)
		SMPL_pred_verts = SMPL_pred_v_homo[:, :, :3, 0] - SMPL_pred_J[:, 0:1, :] + predicted_label_root_xyz.unsqueeze(1)


		# Adjust vertices based on joint addresses by subtracting the joint locations
		SMPL_pred_verts_offset = SMPL_pred_verts.clone().detach()
		predicted_label_markers_xyz_detached = predicted_label_markers_xyz.clone().detach()
		synth_joint_addressed = torch.tensor([3, 15, 4, 5, 7, 8, 18, 19, 20, 21], device=device)
		# Shape: (B, 10, 3) - [(B, 24, 3) -> (B, 10, 3)] -> (B, 10, 3)
		SMPL_pred_verts_offset -= predicted_label_markers_xyz_detached[:, synth_joint_addressed, :]


		# Pad the predicted_labels to increase the second dimension from 91 to 191
		predicted_labels = F.pad(predicted_labels, (0, 100))


		# Update the predicted body_shape (betas) by subtracting the true body_shape (betas)
		predicted_labels[:, 0:10] -= true_labels[:, 72:82]

		predicted_labels[:, 10:16] = predicted_labels[:, 13:19].clone()

		# When loss_type == 'anglesDC'
		predicted_labels[:, 10:13] = predicted_labels[:, 10:13].clone() - torch.cos(true_labels[:, 82:85].clone())	# 1/24 of the true_joint_angles[82:154]
		predicted_labels[:, 13:16] = predicted_labels[:, 13:16].clone() - torch.sin(true_labels[:, 82:85].clone())

		# Compute the scaled difference between true and predicted marker positions
		predicted_labels[:, 40:112] = (true_labels[:, :72] / 1000) - predicted_label_markers_xyz.reshape(-1, 72)
		# Store the squared difference for potential loss calculation or further processing
		predicted_labels[:, 112:184] = torch.square(predicted_labels[:, 40:112].clone() + 1e-7)  # Avoid zero gradients

		predicted_labels = predicted_labels[:, :40]

		# Normalize predicted labels using standard deviations
		betas_std = 1.728158146914805
		body_rot_std = 0.3684988513298487
		joints_std = 0.1752780723422608

		predicted_labels[:, 0:10] /= betas_std			# Normalize betas
		predicted_labels[:, 10:16] /= body_rot_std	# Normalize body rotation
		predicted_labels[:, 16:40] /= joints_std		# Normalize joints

		# return predicted_labels, INPUT_DICT, OUTPUT_DICT
		# def train_convnet(self, epoch):

		# Initialize a tensor of zeros with the same shape as predicted_labels
		scores_zeros = torch.zeros_like(predicted_labels, device=device, requires_grad=True)

		loss_root_rotation = criterion(predicted_labels[:, 10:16], scores_zeros[:, 10:16]) if config['use_root_loss'] else 0.0

		# Calculate the Euclidean loss for joint positions
		loss_eucl = criterion(predicted_labels[:, 16:40], scores_zeros[:, 16:40])

		# Calculate the loss for betas with optional halving
		loss_betas = criterion(predicted_labels[:, :10], scores_zeros[:, :10])
		if config['half_betas_loss']:
			loss_betas *= 0.5

		# Combine the losses
		loss = loss_root_rotation + loss_eucl + loss_betas if config['use_root_loss'] else loss_eucl + loss_betas

		loss.backward()
		optimizer.step()
		loss *= 1000

############################################################################################################################################################################
		if train_batch_index % 10 == 0:
			print(f'Training Summary:')
			print(f'Epoch:				{epoch}/{config["num_epochs"]}')
			print(f'Batch:				{train_batch_index}/{len(train_loader)}')
			print(f'Examples:			{train_batch_index * config["batch_size"]}/{len(train_loader.dataset)}')
			print(f'Loss:				{loss.item():.6f}')
			print(f'Epoch Progress:		{100.0 * train_batch_index / len(train_loader):.2f}%')

			print_error_summary(true_labels[:, :72], predicted_label_markers_xyz_detached)
			print()

			# visualization_image_index = 0

			# tar_sample = true_labels[:, :72].clone()
			# tar_sample = tar_sample[visualization_image_index].squeeze() / 1000
			# sc_sample = predicted_label_markers_xyz_detached.clone()
			# sc_sample = sc_sample[visualization_image_index].squeeze() / 1000
			# sc_sample = sc_sample.view(24, 3)


			# val_n_batches = 4
			# val_loss = evaluate(n_batches=val_n_batches)
			# print(f'Validation Loss: {val_loss:.6f}')

			# # print_text_list = [ 'Train Epoch: {} ',
			# # 					'[{}',
			# # 					'/{} ',
			# # 					'({:.0f}%)]\t']
			# # print_vals_list = [epoch,
			# # 					examples_this_epoch,
			# # 					len(train_loader.dataset),
			# # 					epoch_progress]


def main():
	parser = argparse.ArgumentParser(description='Train PressureNet Model')
	parser.add_argument('--add_noise', type=float, default=0.1, help='amount of noise to add, 0 means no noise')
	parser.add_argument('--include_weight_height', action='store_true', default=False, help='include height and weight as input channels')
	parser.add_argument('--omit_contact_sobel', action='store_true', default=False, help='omit contact sobel')
	parser.add_argument('--use_hover', action='store_true', default=False, help='set depth_map_estimated_positive (channel 1) to 0')
	parser.add_argument('--mod', type=int, choices=[1, 2], required=True, help='choose a network (1 or 2)')
	parser.add_argument('--pmr', action='store_true', default=False, help='run PMR on input & precomputed spatial maps')
	parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
	parser.add_argument('--use_relu', action='store_true', default=False, help='use ReLU in place of Tanh in middle layers of CNN')
	args = parser.parse_args()

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
		'batch_size':			16,
		'num_epochs':			10,
		'learning_rate':		0.00002,
		'normalize_per_image':	True,
		'half_betas_loss':		False,	# Halve the loss for betas
		'use_root_loss':		False,	# Use the root rotation loss
	}

	is_cuda_available = torch.cuda.is_available()
	device = torch.device("cuda" if is_cuda_available else "cpu")

	print(f"Device (CUDA/CPU):  {device}")
	if is_cuda_available:
		print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
		print(f"Device Count:       {torch.cuda.device_count()}")
		print(f"Current Device:     {torch.cuda.current_device()}")
	else:
		print("CUDA is not available, using CPU.")

	# Set the np.array print options
	np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

	# Normalization standard deviations for each channel
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
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
	valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

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


	# 3. Training Loop
	best_valid_loss = float('inf')

	for epoch in range(1, config['num_epochs'] + 1):
		print(f"Epoch [{epoch}/{config['num_epochs']}] started.")

		train(model, train_loader, optimizer, criterion1, device, config, epoch)
		# valid_loss, valid_accuracy = validate(model, valid_loader, criterion2, device, config)
		# print(f"Epoch [{epoch}], Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

		# # Save best model
		# if valid_loss < best_valid_loss:
		# 	best_valid_loss = valid_loss
		# 	torch.save(model.state_dict(), "best_model.pth")
		# 	print("Best model saved!")

		print(f"Epoch [{epoch}/{config['num_epochs']}] completed.")
		# if config['verbose']:
		# 	plot_input_channels(inputs_batch, train_batch_idx)

	# # 7. Load Best Model for Final Testing
	# best_model = PressureNet(in_channels=train_dataset[0][0].shape[0], num_classes=88, use_relu=config['use_relu']).to(device)
	# best_model.load_state_dict(torch.load("best_model.pth"))

if __name__ == '__main__':
	main()
