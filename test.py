#!/usr/bin/env python3
"""
Script to evaluate a trained PressureNetMultiHead model on the test split of the preprocessed HDF5 dataset.
Computes average test loss and MPJPE (mean per-joint position error in mm).

Configure file paths and parameters below without argparse.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets import HDF5Dataset
from models_multi_head import PressureNetMultiHead as PressureNet
from smpl_class import SMPLPreloader
import smplx
from utils_geometry import axis_angle_to_matrix

# -------------------- Configuration --------------------
# Paths
CHECKPOINT_PATH = "/home/nashah/projects/bodies-at-rest/runs/run_20250622_004725/best_model.pth"
HDF5_PATH = "/home/nashah/scratch/data/pre_processed/preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5"

# DataLoader params
BATCH_SIZE = 512
NUM_WORKERS = 4

# Model options
USE_RELU = True

# Normalization stats (same as training)
MEAN = [26.201084, 11.778635, 11.731706]
STD = [41.360558, 27.982226, 8.824089]
# -------------------------------------------------------

def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""
	Mean geodesic distance between predicted and GT rotation matrices on SO(3).
	R_pred, R_gt: (...,3,3)
	"""
	R_rel = R_pred.transpose(-2, -1).matmul(R_gt)
	cos = (R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2] - 1) / 2
	cos = cos.clamp(-1 + eps, 1 - eps)
	theta = torch.acos(cos)
	return theta.mean()


def test(model, test_loader, device, smpl_preloader, config):
	"""
	Evaluate model on test data. Returns average loss and MPJPE (in meters).
	"""
	criterion_joints = nn.MSELoss()
	criterion_betas = nn.MSELoss()
	criterion_transl = nn.MSELoss()

	running_loss = 0.0
	running_mpjpe = 0.0
	total_samples = 0

	model.eval()
	with torch.no_grad():
		for inputs, ground_truth in test_loader:
			inputs = inputs.to(device)
			ground_truth = ground_truth.to(device)
			B = inputs.size(0)
			total_samples += B

			smpl_params_pred = model(inputs)
			betas_pred = smpl_params_pred[:, :10]
			global_orient_aa_pred = smpl_params_pred[:, 10:13]
			body_pose_aa_pred = smpl_params_pred[:, 13:82].reshape(-1, 23, 3)
			transl_pred = smpl_params_pred[:, 82:85]

			joints_pred = smpl_preloader.smpl_forward(
				betas_pred, global_orient_aa_pred,
				body_pose_aa_pred, transl_pred,
				ground_truth
			)  # (B, 24, 3)

			joints_gt = ground_truth[:, :72].reshape(B, 24, 3)
			smpl_params_gt = ground_truth[:, 72:157]
			betas_gt = smpl_params_gt[:, :10]
			global_orient_aa_gt = smpl_params_gt[:, 10:13]
			body_pose_aa_gt = smpl_params_gt[:, 13:82].reshape(-1, 23, 3)
			transl_gt = smpl_params_gt[:, 82:85]

			# Losses
			joints_loss = criterion_joints(joints_pred, joints_gt)
			betas_loss = criterion_betas(betas_pred, betas_gt)
			transl_loss = criterion_transl(transl_pred, transl_gt)

			global_orient_rotMat_pred = axis_angle_to_matrix(global_orient_aa_pred)
			global_orient_rotMat_gt = axis_angle_to_matrix(global_orient_aa_gt)
			global_orient_loss = geodesic_loss(global_orient_rotMat_pred, global_orient_rotMat_gt)

			body_pose_rotMat_pred = axis_angle_to_matrix(body_pose_aa_pred)
			body_pose_rotMat_gt = axis_angle_to_matrix(body_pose_aa_gt)
			body_pose_loss = geodesic_loss(body_pose_rotMat_pred, body_pose_rotMat_gt)

			smpl_params_loss = (
				config['w_betas'] * betas_loss +
				config['w_transl'] * transl_loss +
				config['w_global_orient'] * global_orient_loss +
				config['w_body_pose'] * body_pose_loss
			)

			batch_loss = config['w_joints'] * joints_loss + smpl_params_loss
			running_loss += batch_loss.item() * B

			delta = joints_pred - joints_gt
			per_joint_err = torch.linalg.vector_norm(delta, ord=2, dim=2)
			running_mpjpe += per_joint_err.mean().item() * B

	avg_loss = running_loss / total_samples
	avg_mpjpe = running_mpjpe / total_samples
	return avg_loss, avg_mpjpe


def main():
	# Device setup
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Transforms and DataLoader
	transform = T.Normalize(mean=MEAN, std=STD)
	test_dataset = HDF5Dataset(hdf5_file_path=HDF5_PATH, split='test', transform=transform)
	test_loader = DataLoader(
		test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available()
	)

	# Model
	model = PressureNet(in_channels=test_dataset.num_channels, use_relu=USE_RELU).to(device)
	ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
	state_dict = ckpt.get('model_state_dict', ckpt)
	model.load_state_dict(state_dict)
	print("Loaded checkpoint from", CHECKPOINT_PATH)

	# SMPL Preloader
	smpl_male = smplx.SMPL('/home/nashah/projects/bodies-at-rest/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl').to(device)
	smpl_fem = smplx.SMPL('/home/nashah/projects/bodies-at-rest/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl').to(device)
	config = {
		'w_joints': 1.0,
		'w_betas': 0.1,
		'w_transl': 0.1,
		'w_global_orient': 1.0,
		'w_body_pose': 1.0
	}
	smpl_preloader = SMPLPreloader(smpl_male, smpl_fem, device)

	# Run evaluation
	test_loss, test_mpjpe = test(model, test_loader, device, smpl_preloader, config)
	print(f"Test Loss: {test_loss:.4f} | Test MPJPE: {test_mpjpe * 100:.2f} cm")

if __name__ == '__main__':
	main()