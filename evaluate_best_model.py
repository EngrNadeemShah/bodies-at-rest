# evaluate_best_model.py

import os
import json
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from smplx import SMPL
from models_multi_head import PressureNetMultiHead as PressureNet
from smpl_class import SMPLPreloader
from datasets import HDF5Dataset
from utils_geometry import axis_angle_to_matrix
from utils import get_preprocessed_hdf5_path

def convert_numpy(obj):
	if isinstance(obj, dict):
		return {k: convert_numpy(v) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [convert_numpy(v) for v in obj]
	elif isinstance(obj, tuple):
		return tuple(convert_numpy(v) for v in obj)
	elif isinstance(obj, (np.integer, np.int32, np.int64)):
		return int(obj)
	elif isinstance(obj, (np.floating, np.float32, np.float64)):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	else:
		return obj

def evaluate(model, loader, device, smpl_preloader):
	model.eval()

	# Global containers
	stats = {
		"overall": {"mpjpe_cm": [], "v2v_cm": [], "jointwise_mpjpe_cm": []},
		"male":    {"mpjpe_cm": [], "v2v_cm": [], "jointwise_mpjpe_cm": []},
		"female":  {"mpjpe_cm": [], "v2v_cm": [], "jointwise_mpjpe_cm": []}
	}

	with torch.no_grad():
		for inputs, ground_truth in loader:
			inputs = inputs.to(device)
			ground_truth = ground_truth.to(device)
			B = inputs.shape[0]

			gender_flags = ground_truth[:, 157:159]
			is_female = gender_flags[:, 0].bool()
			is_male   = gender_flags[:, 1].bool()

			smpl_params_pred = model(inputs)

			betas_pred         = smpl_params_pred[:, :10]
			global_orient_pred = smpl_params_pred[:, 10:13]
			body_pose_pred     = smpl_params_pred[:, 13:82].reshape(-1, 23, 3)
			transl_pred        = smpl_params_pred[:, 82:85]

			smpl_gt_params = ground_truth[:, 72:157]
			betas_gt  = smpl_gt_params[:, :10]
			global_gt = smpl_gt_params[:, 10:13]
			pose_gt   = smpl_gt_params[:, 13:82].reshape(-1, 23, 3)
			transl_gt = smpl_gt_params[:, 82:85]

			joints_pred, verts_pred = smpl_preloader.smpl_forward(
				betas_pred, global_orient_pred, body_pose_pred, transl_pred, ground_truth, return_verts=True)
			_, verts_gt = smpl_preloader.smpl_forward(
				betas_gt, global_gt, pose_gt, transl_gt, ground_truth, return_verts=True)

			joints_gt = ground_truth[:, :72].reshape(B, 24, 3)

			joint_errors = torch.norm(joints_pred - joints_gt, dim=2) * 100  # (B, 24)
			mpjpe_batch  = joint_errors.mean(dim=1)                          # (B,)
			v2v_batch    = torch.norm(verts_pred - verts_gt, dim=2).mean(dim=1) * 100  # (B,)

			# Append per-sample stats
			stats["overall"]["mpjpe_cm"].extend(mpjpe_batch.cpu().numpy())
			stats["overall"]["v2v_cm"].extend(v2v_batch.cpu().numpy())
			stats["overall"]["jointwise_mpjpe_cm"].extend(joint_errors.cpu().numpy())

			if is_male.any():
				stats["male"]["mpjpe_cm"].extend(mpjpe_batch[is_male].cpu().numpy())
				stats["male"]["v2v_cm"].extend(v2v_batch[is_male].cpu().numpy())
				stats["male"]["jointwise_mpjpe_cm"].extend(joint_errors[is_male].cpu().numpy())

			if is_female.any():
				stats["female"]["mpjpe_cm"].extend(mpjpe_batch[is_female].cpu().numpy())
				stats["female"]["v2v_cm"].extend(v2v_batch[is_female].cpu().numpy())
				stats["female"]["jointwise_mpjpe_cm"].extend(joint_errors[is_female].cpu().numpy())

	# Compute averages
	result = {}
	for key in ["overall", "male", "female"]:
		mpjpe_vals = np.array(stats[key]["mpjpe_cm"])
		v2v_vals   = np.array(stats[key]["v2v_cm"])
		joint_vals = np.array(stats[key]["jointwise_mpjpe_cm"])  # (N, 24)

		if len(mpjpe_vals) == 0:
			result[key] = { "mpjpe_cm": None, "v2v_cm": None, "jointwise_mpjpe_cm": None }
		else:
			result[key] = {
				"mpjpe_cm": float(mpjpe_vals.mean()),
				"v2v_cm": float(v2v_vals.mean()),
				"jointwise_mpjpe_cm": joint_vals.mean(axis=0).tolist()
			}

	result["per_sample"] = {
		"mpjpe_cm": {
			"overall": stats["overall"]["mpjpe_cm"],
			"male": stats["male"]["mpjpe_cm"],
			"female": stats["female"]["mpjpe_cm"]
		},
		"v2v_cm": {
			"overall": stats["overall"]["v2v_cm"],
			"male": stats["male"]["v2v_cm"],
			"female": stats["female"]["v2v_cm"]
		},
		"jointwise_mpjpe_cm": {
			"overall": stats["overall"]["jointwise_mpjpe_cm"],
			"male": stats["male"]["jointwise_mpjpe_cm"],
			"female": stats["female"]["jointwise_mpjpe_cm"]
		}
	}

	return result

if __name__ == '__main__':
	run_dir = "/home/nashah/projects/bodies-at-rest/runs/20250704_175521"
	config_path = os.path.join(run_dir, "config.json")
	model_path  = os.path.join(run_dir, "best_model.pth")

	with open(config_path, "r") as f:
		config = json.load(f)["config"]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load model
	model = PressureNet(in_channels=3, use_relu=config["training"]["use_relu"]).to(device)
	checkpoint = torch.load(model_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model_state_dict"])

	# Fix SMPL model paths
	base_dir = os.path.dirname(os.path.abspath(__file__))
	config["paths"]["smpl_male_model_path"] = os.path.join(base_dir, config["paths"]["smpl_male_model_path"])
	config["paths"]["smpl_feml_model_path"] = os.path.join(base_dir, config["paths"]["smpl_feml_model_path"])

	smpl_male = SMPL(config["paths"]["smpl_male_model_path"]).to(device)
	smpl_feml = SMPL(config["paths"]["smpl_feml_model_path"]).to(device)
	smpl_preloader = SMPLPreloader(smpl_male, smpl_feml, device)

	transform = T.Normalize(
		mean=[26.201084, 11.778635, 11.731706],
		std=[41.360558, 27.982226, 8.824089]
	)

	hdf5_path = get_preprocessed_hdf5_path(config["paths"]["hdf5_file_name"])
	test_dataset = HDF5Dataset(hdf5_path, split='test', transform=transform)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

	# Evaluate and export
	stats = evaluate(model, test_loader, device, smpl_preloader)
	output_path = os.path.join(run_dir, "final_test_metrics.json")
	stats_serializable = convert_numpy(copy.deepcopy(stats))
	with open(output_path, "w") as f:
		json.dump(stats_serializable, f, indent=2)

	print(f"\n✅ Saved evaluation to: {output_path}")
