import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import sys
from datasets import HDF5Dataset
from models_multi_head import PressureNetMultiHead as PressureNet
from smpl_class import SMPLPreloader
import smplx
from time import time
from utils import print_error_summary, retrieve_data_file_paths, plot_input_channels, format_stats, print_mean_of_model_weights_and_gradients, get_preprocessed_hdf5_path, save_checkpoint
from datetime import datetime
import os
from torch.amp import GradScaler, autocast
from torchinfo import summary
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from utils_geometry import axis_angle_to_matrix
from typing import List
import subprocess, json
import pandas as pd

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda_available else "cpu")


# ——— Load Pre-computed GT standard‐deviations ———
stats_path = "/home/nashah/projects/bodies-at-rest/stats_train_labels_processed_straight_limbs.xlsx"
avg = pd.read_excel(stats_path, sheet_name="avg_straight_limbs", engine="openpyxl")
avg_std = torch.tensor(avg["std_dev"].values, dtype=torch.float32, device=device)

# slice & reshape to match pred-tensor shapes
joints_std        = avg_std[   0:   72].view(1, 24, 3)   # 24 joints × (x,y,z)
betas_std         = avg_std[  72:   82].view(1, 10)      # 10 shape coefs
transl_std        = avg_std[ 154:  157].view(1,  3)      # root translation (x,y,z)

# std_devs of your ground-truth axis-angle magnitudes.
global_orient_geo_std	= 0.149058
body_pose_geo_std		= 0.278674


# --- Loss Functions ---
def standardized_mse(pred, gt, std):
	"""Mean Squared Error loss with standardization by std-dev.
	Standardize each error by its empirical standard-deviation before squaring and averaging.
	That way, every loss term lives in a roughly unit‐variance space and is directly comparable.
	"""
	return torch.mean(((pred - gt) / std)**2)

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

# Loss functions for each output head
criterion_joints = lambda p, g: standardized_mse(p, g, joints_std)
criterion_betas  = lambda p, g: standardized_mse(p, g, betas_std)
criterion_transl = lambda p, g: standardized_mse(p, g, transl_std)

# divide each raw geodesic loss by its train‐set std‐dev
criterion_global_orient = lambda R_pred, R_gt: geodesic_loss(R_pred, R_gt) / global_orient_geo_std
criterion_body_pose     = lambda R_pred, R_gt: geodesic_loss(R_pred, R_gt) / body_pose_geo_std


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


def train(model, train_loader, device, smpl_preloader, CONFIG, adaptive_loss_weights, optimizer, scaler):
	# torch.autograd.set_detect_anomaly(True)
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
		global_orient_loss = criterion_global_orient(global_orient_rotMat_pred, global_orient_rotMat_gt)

		# Body pose (axis-angle → Rotation matrices)
		body_pose_rotMat_pred = axis_angle_to_matrix(body_pose_aa_pred)	# (B, 23, 3) -> (B, 23, 3, 3)
		body_pose_rotMat_gt   = axis_angle_to_matrix(body_pose_aa_gt)	# (B, 23, 3) -> (B, 23, 3, 3)
		body_pose_loss = criterion_body_pose(body_pose_rotMat_pred, body_pose_rotMat_gt)

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

		# Prevent log_vars from running away
		with torch.no_grad():
			adaptive_loss_weights.log_vars.data.clamp_(min=-5.0, max=+5.0)

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

def validate(model, valid_loader, device, smpl_preloader, CONFIG, adaptive_loss_weights):
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
				global_orient_loss = criterion_global_orient(global_orient_rotMat_pred, global_orient_rotMat_gt)

				# Body pose (axis-angle → Rotation matrices)
				body_pose_rotMat_pred = axis_angle_to_matrix(body_pose_aa_pred)	# (B, 23, 3) -> (B, 23, 3, 3)
				body_pose_rotMat_gt   = axis_angle_to_matrix(body_pose_aa_gt)	# (B, 23, 3) -> (B, 23, 3, 3)
				body_pose_loss = criterion_body_pose(body_pose_rotMat_pred, body_pose_rotMat_gt)

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
	start_time = time()
	# 0. Initializations and Configurations
	# Set the configuration parameters
	CONFIG = {
		"run": {
			"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
			"verbose": False,
			"checkpoint": {
				"save_every_epochs": 5,
				"early_stopping_patience": 10,
				"log_hist_every": 5,
			},
		},

		"paths": {
			"run_dir": '',
			"smpl_male_model_path": "smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
			"smpl_feml_model_path": "smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl",
			# "hdf5_file_name": "preprocessed_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5",
			# "hdf5_file_name": "preprocessed_mod2_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_2__normalize_per_image_True.hdf5",
			# "hdf5_file_name": "preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True_no_75mm.hdf5",
			"hdf5_file_name": "preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5",
		},

		"training": {
			"batch_size": 512,
			"num_epochs": 100,
			"use_relu": True
		},

		"optimizer": {
			"type": "AdamW",
			"lr_init": 1e-3,		# tune: 1e-4 to 1e-3 -> (default: 1e-4)
			"weight_decay": 5e-4,	# L2 regularization | tune: 1e-5 to 1e-2 -> (default: 5e-4)
			"scheduler": {
				"type": "CosineAnnealingLR",
				"eta_min": 1e-6,	# LR floor for CosineAnnealing | tune: 0 → 1e-5 -> (default: 1e-6)
			},
		},

		"dataloader": {
			"pin_memory": True,
			"train_workers": 28,
			"valid_workers": 28,
			"prefetch_train": 2,
			"prefetch_valid": 2,
			"persistent_workers_train": True,
			"persistent_workers_valid": True,
			},

		"hardware": {},
		}

	# Define the hyperparameters which you want to track in TensorBoard
	HYPERPARAMS = {
		"lr_init":      CONFIG["optimizer"]["lr_init"],
		"weight_decay": CONFIG["optimizer"]["weight_decay"],
		"batch_size":   CONFIG["training"]["batch_size"],
		"use_relu":     CONFIG["training"]["use_relu"],
	}

	# Print the device information
	if is_cuda_available:
		CONFIG["hardware"]["gpu_available"]	= True
		CONFIG["hardware"]["gpu_count"]		= torch.cuda.device_count()
		CONFIG["hardware"]["gpu_name"]		= torch.cuda.get_device_name(0)
		CONFIG["hardware"]["gpu_devices"] = [
			torch.cuda.get_device_name(i)
			for i in range(torch.cuda.device_count())]
		CONFIG["hardware"]["current_device"]= torch.cuda.current_device()
		CONFIG["hardware"]["cpu_count"] = os.cpu_count()
	else:
		CONFIG["hardware"]["gpu_available"]	= False
		CONFIG["hardware"]["gpu_count"]		= 0
		CONFIG["hardware"]["gpu_name"]		= "N/A"
		CONFIG["hardware"]["current_device"]= "cpu"
		CONFIG["hardware"]["cpu_count"] = os.cpu_count()
	print(f"Device (CUDA/CPU):  {device}")

	# Create the run directory and the path for HDF5 file
	CONFIG["paths"]["run_dir"] = os.path.join("runs", CONFIG["run"]["timestamp"])
	os.makedirs(CONFIG["paths"]["run_dir"], exist_ok=True)
	hdf5_file_path = get_preprocessed_hdf5_path(CONFIG["paths"]["hdf5_file_name"])

	# Save the configuration to a JSON file
	config_dump = {
		"config": CONFIG,
		# "git_commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
		"torch_version": torch.__version__,
		"python_version": sys.version.split()[0],
	}
	with open(os.path.join(CONFIG["paths"]["run_dir"], "config.json"), "w") as f:
		json.dump(config_dump, f, indent=2)

	# Initialize TensorBoard writer
	writer = SummaryWriter(log_dir=CONFIG["paths"]["run_dir"])


	# 1. Data Preparation

	# Create the train and valid datasets and data loaders

	# Prepare the transforms for the dataset
	# Caclulated on train_straight_limbs_hdf5_mod1_input_images for all 3 channels (for male & female, then averaged)
	transform = T.Normalize(
		mean=[26.201084, 11.778635, 11.731706],
		std	=[41.360558, 27.982226, 8.824089])

	train_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='train', transform=transform)
	valid_dataset = HDF5Dataset(hdf5_file_path=hdf5_file_path, split='test', transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True,	num_workers=CONFIG['dataloader']['train_workers'], pin_memory=CONFIG['dataloader']['pin_memory'], prefetch_factor=CONFIG['dataloader']['prefetch_train'], persistent_workers=CONFIG['dataloader']['persistent_workers_train'])
	valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=False,num_workers=CONFIG['dataloader']['valid_workers'], pin_memory=CONFIG['dataloader']['pin_memory'], prefetch_factor=CONFIG['dataloader']['prefetch_valid'], persistent_workers=CONFIG['dataloader']['persistent_workers_valid'])


	# 2. Define the model, optimizer, and loss functions
	model = PressureNet(in_channels=train_dataset.num_channels, use_relu=CONFIG['training']['use_relu']).to(device)

	# Logging the model graph
	dummy = torch.zeros(
		(1, train_dataset.num_channels, 128, 54),
		device=device,
		dtype=torch.float32)
	writer.add_graph(model, (dummy,))


	# — adaptive weights for joint & SMPL losses —
	adaptive_loss_weights = AdaptiveLoss().to(device)
	# Initialize log_vars to sensible priors, e.g. if you want all wᵢ=1 except betas=0.1 at start:
	init_ws = torch.tensor([1.0, 0.1, 1.0, 1.0, 1.0], device=device)
	# we want exp(-log_var) = w  =>  log_var = -log(w)
	adaptive_loss_weights.log_vars.data = -torch.log(init_ws)

	optimizer = AdamW(list(model.parameters()) + list(adaptive_loss_weights.parameters()),
        lr=CONFIG['optimizer']['lr_init'], weight_decay=CONFIG['optimizer']['weight_decay'])

	# Decay LR from lr_init → eta_min over 'num_epochs'
	scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['training']['num_epochs'], eta_min=CONFIG['optimizer']['scheduler']['eta_min'])

	if CONFIG['run']['verbose']:
		print("\nModel Summary:")
		print(model)
		print()
		summary(model, input_size=(CONFIG['training']['batch_size'], train_dataset.num_channels, 128, 54), device=device.type)


	# 3. Load SMPL models
	smpl_male_model = smplx.SMPL(CONFIG["paths"]["smpl_male_model_path"]).to(device)
	smpl_feml_model = smplx.SMPL(CONFIG["paths"]["smpl_feml_model_path"]).to(device)


	# 4. Training Loop
	best_valid_loss		= float('inf')
	best_valid_epoch	= -1
	best_train_loss		= None
	best_train_mpjpe	= None
	best_valid_mpjpe	= None
	epochs_without_improvement = 0	# counter for early stopping

	train_valid_losses = {
		'epoch': [],
		'train_loss': [],
		'valid_loss': [],
	}

	scaler = GradScaler(device=device.type)

	try:
		for epoch in range(1, CONFIG['training']['num_epochs'] + 1):
			print("*" * 50)
			print(f"Epoch: {epoch:03d}/{CONFIG['training']['num_epochs']:03d}")
			print("*" * 50)

			# Initialize preloader before training loop (once per epoch)
			smpl_preloader = SMPLPreloader(smpl_male_model, smpl_feml_model, device)

			print("-" * 30)
			train_loss, train_mpjpe = train(model, train_loader, device, smpl_preloader, CONFIG, adaptive_loss_weights, optimizer, scaler)
			print(f"Training (Epoch {epoch:03d}) - Loss: {train_loss:.4f} | MPJPE: {train_mpjpe*1000:.4f} mm")
			print("-" * 30)

			print("=" * 30)
			valid_loss, valid_mpjpe = validate(model, valid_loader, device, smpl_preloader, CONFIG, adaptive_loss_weights)
			print(f"Validation (Epoch {epoch:03d}) - Loss: {valid_loss:.4f} | MPJPE: {valid_mpjpe*1000:.4f} mm")
			print("=" * 30)

			# Update the learning rate
			scheduler.step()

			# Print the current learning rate
			current_lr = scheduler.get_last_lr()[0]
			print(f"Epoch {epoch:03d} - lr: {current_lr:.2e} ({current_lr:.6f}) ({CONFIG['optimizer']['lr_init']:.2e} → {CONFIG['optimizer']['scheduler']['eta_min']:.2e})")

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
			if epoch % CONFIG['run']['checkpoint']['log_hist_every'] == 0:
				for name, param in model.named_parameters():
					# 1) log their weights
					writer.add_histogram(name, param, epoch)
					# 2) log their gradients (if computed)
					if param.grad is not None:
						writer.add_histogram(f"{name}.grad", param.grad, epoch)

			# Best‐model checkpoint
			if valid_loss < best_valid_loss:
				best_valid_epoch			= epoch
				best_valid_loss		= valid_loss
				best_train_loss		= train_loss
				best_train_mpjpe	= train_mpjpe
				best_valid_mpjpe	= valid_mpjpe
				epochs_without_improvement = 0
				save_checkpoint(
					os.path.join(CONFIG["paths"]["run_dir"], 'best_model.pth'),
					epoch, model, optimizer, scheduler,
					train_valid_losses, best_valid_loss, scaler)
			else:
				epochs_without_improvement += 1
				print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

			# Early stopping if no improvement in validation loss for `early_stopping_patience` epochs
			if epochs_without_improvement >= CONFIG['run']['checkpoint']['early_stopping_patience']:
				print(f"Stopping early at epoch {epoch} after {epochs_without_improvement} epochs with no improvement.")
				break

			# Periodic checkpoint
			if epoch % CONFIG['run']['checkpoint']['save_every_epochs'] == 0 or epoch == CONFIG['training']['num_epochs']:
				save_checkpoint(
					os.path.join(CONFIG["paths"]["run_dir"], f'ckpt_epoch{epoch:03d}_vloss{valid_loss:.4f}.pth'),
					epoch, model, optimizer, scheduler,
					train_valid_losses, best_valid_loss, scaler)

	finally:
		# Write out a simple results.json for downstream scripts
		results = {
			"best_valid_epoch":		best_valid_epoch,
			"best_valid_loss":		best_valid_loss,
			"train_loss_at_best":	best_train_loss,
			'train_mpjpe_at_best':	best_train_mpjpe,
			'valid_mpjpe_at_best':  best_valid_mpjpe,
			"total_time_s":          time() - start_time
		}
		results_path = os.path.join(CONFIG["paths"]["run_dir"], "results.json")
		with open(results_path, "w") as f:
			json.dump(results, f, indent=2)

		# Build your metrics dict
		metrics_dict = {
			'best_valid_epoch':   float(best_valid_epoch),
			'best_valid_loss':     best_valid_loss,
			'train_loss_at_best':   best_train_loss,
			'train_mpjpe_at_best':  best_train_mpjpe,
			'valid_mpjpe_at_best':  best_valid_mpjpe,
		}

		# One‐shot hparams write
		writer.add_hparams(HYPERPARAMS, metrics_dict)
		writer.flush()
		writer.close()

if __name__ == '__main__':
	main()
