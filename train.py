import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from datasets import HDF5Dataset
from models_multi_head import PressureNetMultiHead as PressureNet
from smpl_class import SMPLPreloader
import smplx
from time import time
from tqdm import tqdm
import pickle as pkl
from utils import print_error_summary, retrieve_data_file_paths, plot_input_channels, format_stats, print_mean_of_model_weights_and_gradients, get_preprocessed_hdf5_path, save_checkpoint
from datetime import datetime
import os
from torch.amp import GradScaler, autocast
from torchinfo import summary
from torch.utils.data import Subset
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from utils_geometry import axis_angle_to_matrix

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

# --- Loss Functions ---

# joint‐space loss (MPJPE)
criterion_joints = nn.MSELoss()

# shape (betas) and translation losses
criterion_betas = nn.MSELoss()
criterion_transl = nn.MSELoss()

# geodesic rotation loss on SO(3)
def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""
	Mean geodesic distance between predicted and GT rotation matrices.
	R_pred, R_gt: (...,3,3)
	"""
	R_rel = R_pred.transpose(-2, -1).matmul(R_gt)
	cos = (R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2] - 1) / 2
	cos = cos.clamp(-1 + eps, 1 - eps)
	theta = torch.acos(cos)		# θ in [0, π]
	return theta.mean()

# Replace static w_* (weights for losses) scalars with learnable weights using homoscedastic uncertainty (log-variance) trick
class AdaptiveLoss(nn.Module):
	"""
	Learns a log-variance per loss term; at forward we do:
		total = Σ_i [ exp(–log_vars[i]) * L_i  +  log_vars[i] ]
	so that exp(–log_vars[i]) is the effective weight on L_i.
	"""
	def __init__(self):
		super().__init__()
		# 5 terms: joints, betas, global_orient, body_pose, transl
		self.log_vars = nn.Parameter(torch.zeros(5))

	def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
		total = 0.0
		for i, Li in enumerate(losses):
			inv_var = torch.exp(-self.log_vars[i])
			total += inv_var * Li + self.log_vars[i]
		return total


def train(model, train_loader, device, smpl_preloader, config, adaptive_loss_weights, optimizer, scaler):
	torch.autograd.set_detect_anomaly(True)
	print(f"\nTraining ...")

	running_loss = 0.0
	running_mpjpe = 0.0
	total_samples = 0

	model.train()
	for idx, (inputs, ground_truth) in enumerate(train_loader, 1):
		print(f"Batch: {idx:02}/{len(train_loader):02}", end='\t')

		inputs, ground_truth = inputs.to(device), ground_truth.to(device)

		B = inputs.shape[0]
		total_samples += B

		optimizer.zero_grad()	# Clear gradients

		# — Forward —
		with autocast(device_type=device.type):
			smpl_params_pred = model(inputs)	# (B, 85)

			# Unpack preds
			betas_pred				= smpl_params_pred[:, :10]						# (B, 10)
			global_orient_aa_pred	= smpl_params_pred[:, 10:13]					# (B, 3)
			body_pose_aa_pred		= smpl_params_pred[:, 13:82].reshape(-1, 23, 3)	# (B, 69) -> (B, 23, 3)
			transl_pred				= smpl_params_pred[:, 82:85]					# (B, 3)

			# Pass the CNN model output through SMPL model to get the joint positions
			joints_pred = smpl_preloader.smpl_forward(betas_pred, global_orient_aa_pred, body_pose_aa_pred, transl_pred, ground_truth)	# (B, 24, 3)

			# Unpack GT
			joints_gt			= ground_truth[:, :72].reshape(B, 24, 3)		# (B, 24, 3)
			smpl_params_gt		= ground_truth[:, 72:157]						# (B, 85)

			betas_gt			= smpl_params_gt[:, :10]						# (B, 10)
			global_orient_aa_gt	= smpl_params_gt[:, 10:13]						# (B, 3)
			body_pose_aa_gt		= smpl_params_gt[:, 13:82].reshape(-1, 23, 3)	# (B, 69) -> (B, 23, 3)
			transl_gt			= smpl_params_gt[:, 82:85]						# (B, 3)

		# — Losses —
		# Joint positions (MPJPE), betas, translation
		joints_loss	= criterion_joints(joints_pred, joints_gt)
		betas_loss	= criterion_betas(betas_pred, betas_gt)
		transl_loss	= criterion_transl(transl_pred, transl_gt)

		# Global orientation (axis-angle → Rotation matrix)
		global_orient_rotMat_pred	= axis_angle_to_matrix(global_orient_aa_pred)	# (B, 3) -> (B, 3, 3)
		global_orient_rotMat_gt		= axis_angle_to_matrix(global_orient_aa_gt)		# (B, 3) -> (B, 3, 3)
		global_orient_loss = geodesic_loss(global_orient_rotMat_pred, global_orient_rotMat_gt)

		# Body pose (axis-angle → Rotation matrices)
		body_pose_rotMat_pred = axis_angle_to_matrix(body_pose_aa_pred)	# (B, 23, 3) -> (B, 23, 3, 3)
		body_pose_rotMat_gt   = axis_angle_to_matrix(body_pose_aa_gt)	# (B, 23, 3) -> (B, 23, 3, 3)
		body_pose_loss = geodesic_loss(body_pose_rotMat_pred, body_pose_rotMat_gt)

		# pack losses in the fixed order matching log_vars:
		# [ joints, betas, global_orient, body_pose, transl ]
		losses = [joints_loss, betas_loss, global_orient_loss, body_pose_loss, transl_loss]
		batch_loss = adaptive_loss_weights(losses)

		# Backward pass and optimization (Outside the autocast context)
		scaler.scale(batch_loss).backward()			# compute (scaled) grads
		scaler.unscale_(optimizer)					# bring them back to real scale
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)	# clip whole-model norm
		scaler.step(optimizer)						# apply the clipped step
		scaler.update()								# update the scale for next iter

		# Accumulate loss
		running_loss += batch_loss.item() * B

		# Compute MPJPE per batch
		delta = joints_pred - joints_gt									# (B, 24, 3)
		per_joint_err = torch.linalg.vector_norm(delta, ord=2, dim=2)	# (B, 24)
		batch_mpjpe = per_joint_err.mean()								# scalar (m)
		running_mpjpe += batch_mpjpe.item() * B

		# Print stats
		print(f"Joints: {joints_loss.item():.4f}", end='\t')
		print(f"Betas: {betas_loss.item():.4f}", end='\t')
		print(f"Transl: {transl_loss.item():.4f}", end='\t')
		print(f"Global: {global_orient_loss.item():.4f}", end='\t')
		print(f"Body: {body_pose_loss.item():.4f}", end='\t')
		print(f"Total: {batch_loss.item():.4f}", end='\t')
		print(f"MPJPE: {batch_mpjpe.item()*1000:.1f} mm", end='\n')

	avg_loss = running_loss / total_samples
	avg_mpjpe = running_mpjpe / total_samples
	return avg_loss, avg_mpjpe

def validate(model, valid_loader, device, smpl_preloader, config, adaptive_loss_weights):
	print(f"\nValidating ...")

	running_loss = 0.0
	running_mpjpe = 0.0
	total_samples = 0

	model.eval()
	with torch.no_grad():
		for idx, (inputs, ground_truth) in enumerate(valid_loader, 1):
			print(f"Batch: {idx:02}/{len(valid_loader):02}", end='\t')

			inputs, ground_truth = inputs.to(device), ground_truth.to(device)
			B = inputs.shape[0]
			total_samples += B

			# — Forward —
			with autocast(device_type=device.type):
				smpl_params_pred = model(inputs)	# (B, 85)

				# Unpack preds
				betas_pred				= smpl_params_pred[:, :10]						# (B, 10)
				global_orient_aa_pred	= smpl_params_pred[:, 10:13]					# (B, 3)
				body_pose_aa_pred		= smpl_params_pred[:, 13:82].reshape(-1, 23, 3)	# (B, 69) -> (B, 23, 3)
				transl_pred				= smpl_params_pred[:, 82:85]					# (B, 3)

				# Pass the CNN model output through SMPL model to get the joint positions
				joints_pred = smpl_preloader.smpl_forward(betas_pred, global_orient_aa_pred, body_pose_aa_pred, transl_pred, ground_truth)	# (B, 24, 3)

				# Unpack GT
				joints_gt			= ground_truth[:, :72].reshape(B, 24, 3)		# (B, 24, 3)
				smpl_params_gt		= ground_truth[:, 72:157]						# (B, 85)

				betas_gt			= smpl_params_gt[:, :10]						# (B, 10)
				global_orient_aa_gt	= smpl_params_gt[:, 10:13]						# (B, 3)
				body_pose_aa_gt		= smpl_params_gt[:, 13:82].reshape(-1, 23, 3)	# (B, 69) -> (B, 23, 3)
				transl_gt			= smpl_params_gt[:, 82:85]						# (B, 3)

				# — Losses —
				# Joint positions (MPJPE), betas, translation
				joints_loss	= criterion_joints(joints_pred, joints_gt)
				betas_loss	= criterion_betas(betas_pred, betas_gt)
				transl_loss	= criterion_transl(transl_pred, transl_gt)

				# Global orientation (axis-angle → Rotation matrix)
				global_orient_rotMat_pred	= axis_angle_to_matrix(global_orient_aa_pred)	# (B, 3) -> (B, 3, 3)
				global_orient_rotMat_gt		= axis_angle_to_matrix(global_orient_aa_gt)		# (B, 3) -> (B, 3, 3)
				global_orient_loss = geodesic_loss(global_orient_rotMat_pred, global_orient_rotMat_gt)

				# Body pose (axis-angle → Rotation matrices)
				body_pose_rotMat_pred = axis_angle_to_matrix(body_pose_aa_pred)	# (B, 23, 3) -> (B, 23, 3, 3)
				body_pose_rotMat_gt   = axis_angle_to_matrix(body_pose_aa_gt)	# (B, 23, 3) -> (B, 23, 3, 3)
				body_pose_loss = geodesic_loss(body_pose_rotMat_pred, body_pose_rotMat_gt)

				# pack losses in the fixed order matching log_vars:
				# [ joints, betas, global_orient, body_pose, transl ]
				losses = [joints_loss, betas_loss, global_orient_loss, body_pose_loss, transl_loss]
				batch_loss = adaptive_loss_weights(losses)

			# Accumulate loss
			running_loss += batch_loss.item() * B

			# Compute MPJPE per batch
			delta = joints_pred - joints_gt									# (B, 24, 3)
			per_joint_err = torch.linalg.vector_norm(delta, ord=2, dim=2)	# (B, 24)
			batch_mpjpe = per_joint_err.mean()								# scalar (m)
			running_mpjpe += batch_mpjpe.item() * B

			# Print stats
			print(f"Loss: {batch_loss.item():.4f}",
				  f"MPJPE: {batch_mpjpe.item()*1000:.1f} mm")

	# Calculate per‐epoch averages
	avg_loss = running_loss / total_samples
	avg_mpjpe = running_mpjpe / total_samples
	return avg_loss, avg_mpjpe

def main():
	# 0. Initializations and Configurations
	# 0.1. Parse the command line arguments
	# parser = argparse.ArgumentParser(description='Train PressureNet Model')
	# parser.add_argument('--mod', type=int, choices=[1, 2], required=True, help='choose a network (1 or 2)')
	# parser.add_argument('--pmr', action='store_true', default=False, help='run PMR on input & precomputed spatial maps')
	# parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
	# parser.add_argument('--use_relu', action='store_true', default=False, help='use ReLU in place of Tanh in middle layers of CNN')
	# args = parser.parse_args()

	# 0.2. Set the configuration parameters
	config = {
		'mod':					1,		# args.mod,
		'pmr':					False,	# args.pmr,
		'verbose':				False,	# args.verbose,
		'use_relu':				True,	# args.use_relu,

		# Not in args
		'batch_size':			512,
		'num_epochs':			100,
		'save_model_every':		5,
		'log_hist_every':		5,
		'early_stop_patience':	10,

		# Hyperparameters (tune within these ranges; defaults in parentheses)
		'lr_init':				0.001,	# 1e-4(0.0001) to 1e-3(0.001)						(default: 3e-4(0.0003))
		'weight_decay':  		0.0005,	# L2 regularization: 1e-5(0.00001) to 1e-2(0.01)	(default: 5e-4(0.0005))
		'eta_min':				1e-6,	# LR floor for CosineAnnealing: 0 to 1e-5(0.00001)	(default: 1e-6(0.000001))

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

	# 0.6. Initialize TensorBoard writer
	writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb_logs'))

	# 1. Data Preparation

	# Create the train and valid datasets and data loaders
	# HDF5 file paths
	# hdf5_file_name = 'preprocessed_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
	# hdf5_file_name = 'preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5'
	# hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True_no_75mm.hdf5'
	hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'

	hdf5_file_path = get_preprocessed_hdf5_path(hdf5_file_name)

	# Prepare the transforms for the dataset
	# Caclulated on train_straight_limbs_hdf5_mod1_input_images for all 3 channels (for male & female, then averaged)
	transform = T.Normalize(
		mean=[26.201084, 11.778635, 11.731706],
		std	=[41.360558, 27.982226, 8.824089])

	train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train', transform=transform)
	valid_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='test', transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,	num_workers=config['num_workers_train'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor_train'], persistent_workers=config['persistent_workers_train'])
	valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False,num_workers=config['num_workers_valid'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor_valid'], persistent_workers=config['persistent_workers_valid'])

	print(f"Learning Rate (Initial):	{config['lr_init']}")
	print(f"Weight Decay:			{config['weight_decay']}")
	print(f"Eta Min (Cosine Annealing):	{config['eta_min']}")
	print(f"Use ReLU:				{config['use_relu']}")
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
	model = PressureNet(in_channels=train_dataset.num_channels, use_relu=config['use_relu']).to(device)


	# — adaptive weights for joint & SMPL losses —
	adaptive_loss_weights = AdaptiveLoss().to(device)
	# init_ws = torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1], device=device)
	# # we want exp(-log_var) = w  =>  log_var = -log(w)
	# adaptive_loss_weights.log_vars.data = -torch.log(init_ws)

	optimizer = AdamW(list(model.parameters()) + list(adaptive_loss_weights.parameters()),
        lr=config['lr_init'], weight_decay=config['weight_decay'])

	# Decay LR from lr_init → eta_min over 'num_epochs'
	scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=config['eta_min'])

	if config['verbose']:
		print("\nModel Summary:")
		print(model)
		print()
		summary(model, input_size=(config['batch_size'], train_dataset.num_channels, 128, 54), device=device.type)


	# 3. Load SMPL models
	smpl_male_model_path = 'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
	smpl_feml_model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

	smpl_male_model = smplx.SMPL(smpl_male_model_path).to(device)
	smpl_feml_model = smplx.SMPL(smpl_feml_model_path).to(device)


	# 4. Training Loop
	best_valid_loss = float('inf')
	no_improve_epochs = 0	# counter for early stopping

	train_valid_losses = {
		'epoch': [],
		'train_loss': [],
		'valid_loss': [],
	}

	scaler = GradScaler(device=device.type)

	for epoch in range(1, config['num_epochs'] + 1):
		print("*" * 50)
		print(f"Epoch: {epoch:03d}/{config['num_epochs']:03d}")
		print("*" * 50)

		# Initialize preloader before training loop (once per epoch)
		smpl_preloader = SMPLPreloader(smpl_male_model, smpl_feml_model, device)

		print("-" * 30)
		train_loss, train_mpjpe = train(model, train_loader, device, smpl_preloader, config, adaptive_loss_weights, optimizer, scaler)
		print(f"Training (Epoch {epoch:03d}) - Loss: {train_loss:.4f} | MPJPE: {train_mpjpe*1000:.4f} mm")
		print("-" * 30)

		print("=" * 30)
		valid_loss, valid_mpjpe = validate(model, valid_loader, device, smpl_preloader, config, adaptive_loss_weights)
		print(f"Validation (Epoch {epoch:03d}) - Loss: {valid_loss:.4f} | MPJPE: {valid_mpjpe*1000:.4f} mm")
		print("=" * 30)

		# Update the learning rate
		scheduler.step()

		# Print the current learning rate
		current_lr = scheduler.get_last_lr()[0]
		print(f"Epoch {epoch:03d} - lr: {current_lr:.2e} ({current_lr:.6f}) ({config['lr_init']:.2e} → {config['eta_min']:.2e})")

		# Get the 5 weights as a CPU tensor and print them
		w_eff = torch.exp(-adaptive_loss_weights.log_vars.data).cpu().tolist()
		print(f"[Epoch {epoch:3d}] Loss weights:"
				f" joints={w_eff[0]:.3f},"
				f" betas={w_eff[1]:.3f},"
				f" global_orient={w_eff[2]:.3f},"
				f" body_pose={w_eff[3]:.3f},"
				f" transl={w_eff[4]:.3f}")

		# Log the loss weights to TensorBoard
		writer.add_scalars('LossWeights', {
			'joints': w_eff[0],
			'betas': w_eff[1],
			'global_orient': w_eff[2],
			'body_pose': w_eff[3],
			'transl': w_eff[4]
		}, epoch)

		# Log the losses, MPJPE, and learning rate to TensorBoard
		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('Loss/valid', valid_loss, epoch)
		writer.add_scalar('MPJPE/train', train_mpjpe*1000, epoch)
		writer.add_scalar('MPJPE/valid', valid_mpjpe*1000, epoch)
		writer.add_scalar('LR', current_lr, epoch)

		# Save the losses in a dictionary
		train_valid_losses['epoch'].append(epoch)
		train_valid_losses['train_loss'].append(train_loss)
		train_valid_losses['valid_loss'].append(valid_loss)

		# Log weight & bias histograms every `log_hist_every` epochs
		if epoch % config['log_hist_every'] == 0:
			for name, param in model.named_parameters():
				writer.add_histogram(name, param, epoch)

		# Best‐model checkpoint
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			no_improve_epochs = 0
			save_checkpoint(
				os.path.join(run_dir, 'best_model.pth'),
				epoch, model, optimizer, scheduler,
				train_valid_losses, best_valid_loss, scaler)
		else:
			no_improve_epochs += 1
			print(f"No improvement in validation loss for {no_improve_epochs} epochs.")

		# Early stopping if no improvement in validation loss for `early_stopping_patience` epochs
		if no_improve_epochs >= config['early_stopping_patience']:
			print(f"Stopping early at epoch {epoch} after {no_improve_epochs} epochs with no improvement.")
			break

		# Periodic checkpoint
		if epoch % config['save_model_every'] == 0 or epoch == config['num_epochs']:
			save_checkpoint(
				os.path.join(run_dir, f'ckpt_epoch{epoch:03d}_vloss{valid_loss:.4f}.pth'),
				epoch, model, optimizer, scheduler,
				train_valid_losses, best_valid_loss, scaler)

if __name__ == '__main__':
	main()
