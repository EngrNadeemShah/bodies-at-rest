import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from smplx import SMPL
from torchvision import transforms
from torch.utils.data import DataLoader

from models_multi_head import PressureNetMultiHead as PressureNet
from smpl_class import SMPLPreloader
from datasets import HDF5Dataset
from utils import get_preprocessed_hdf5_path
from utils_geometry import axis_angle_to_matrix
from render_utils import render_smpl_comparison

def main():
	run_dir = "/home/nadeemshah/coding/bodies-at-rest/runs/20250704_175521"
	config_path = os.path.join(run_dir, "config.json")
	model_path = os.path.join(run_dir, "best_model.pth")
	output_dir = os.path.join(run_dir, "visualizations")
	os.makedirs(output_dir, exist_ok=True)

	# Load config
	with open(config_path, "r") as f:
		config = json.load(f)["config"]

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load model
	model = PressureNet(in_channels=3, use_relu=config["training"]["use_relu"]).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
	model.eval()

	# Load SMPL models
	base = os.path.dirname(os.path.abspath(__file__))
	config["paths"]["smpl_male_model_path"] = os.path.join(base, config["paths"]["smpl_male_model_path"])
	config["paths"]["smpl_feml_model_path"] = os.path.join(base, config["paths"]["smpl_feml_model_path"])
	smpl_m = SMPL(config["paths"]["smpl_male_model_path"]).to(device)
	smpl_f = SMPL(config["paths"]["smpl_feml_model_path"]).to(device)
	smpl_preloader = SMPLPreloader(smpl_m, smpl_f, device)

	# Data
	transform = transforms.Normalize(
		mean=[26.201084, 11.778635, 11.731706],
		std=[41.360558, 27.982226, 8.824089]
	)
	hdf5_path = get_preprocessed_hdf5_path(config["paths"]["hdf5_file_name"])
	test_dataset = HDF5Dataset(hdf5_path, split='test', transform=transform)
	loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	# Collect errors and genders
	results = []
	features = []
	for idx, (x, gt) in enumerate(tqdm(loader, desc="Running Inference")):
		x = x.to(device)
		gt = gt.to(device)
		pred = model(x)

		betas_pred = pred[:, :10]
		global_orient_pred = pred[:, 10:13].unsqueeze(1)
		body_pose_pred = pred[:, 13:82].reshape(1, 23, 3)
		transl_pred = pred[:, 82:85]

		smpl_gt_params = gt[:, 72:157]
		betas_gt = smpl_gt_params[:, :10]
		global_gt = smpl_gt_params[:, 10:13].unsqueeze(1)
		pose_gt = smpl_gt_params[:, 13:82].reshape(1, 23, 3)
		transl_gt = smpl_gt_params[:, 82:85]
		joints_gt = gt[:, :72].reshape(1, 24, 3)

		# Run SMPL forward for predicted joints
		joints_pred = smpl_preloader.smpl_forward(
			betas_pred, global_orient_pred.squeeze(1), body_pose_pred, transl_pred, gt
		)

		mpjpe = torch.norm(joints_pred - joints_gt, dim=2).mean().item()
		is_male = gt[:, 158].item() > 0.5
		is_female = not is_male

		results.append({
			"idx": idx,
			"mpjpe": mpjpe,
			"is_male": is_male,
			"is_female": is_female
		})

		# Save tensors for later rendering
		features.append({
			"x": x,
			"gt": gt,
			"pred": pred,
			"joints_pred": joints_pred,
			"joints_gt": joints_gt
		})

	# Now convert to DataFrame for ranking
	df = pd.DataFrame(results)

	# Pick 3 best and 3 worst per gender
	selected_df = pd.concat([
		df[df["is_male"]].nsmallest(3, "mpjpe").assign(label="best_male"),
		df[df["is_male"]].nlargest(3, "mpjpe").assign(label="worst_male"),
		df[df["is_female"]].nsmallest(3, "mpjpe").assign(label="best_female"),
		df[df["is_female"]].nlargest(3, "mpjpe").assign(label="worst_female")
	])

	faces = smpl_m.faces  # same for both genders

	# Render selected examples
	for _, row in selected_df.iterrows():
		idx = int(row["idx"])
		label = row["label"]
		data = features[idx]

		x = data["x"]
		gt = data["gt"]
		pred = data["pred"]
		joints_pred = data["joints_pred"]
		joints_gt = data["joints_gt"]

		betas_pred = pred[:, :10]
		global_orient_pred = pred[:, 10:13].unsqueeze(1)
		body_pose_pred = pred[:, 13:82].reshape(1, 23, 3)
		transl_pred = pred[:, 82:85]

		smpl_gt_params = gt[:, 72:157]
		betas_gt = smpl_gt_params[:, :10]
		global_gt = smpl_gt_params[:, 10:13].unsqueeze(1)
		pose_gt = smpl_gt_params[:, 13:82].reshape(1, 23, 3)
		transl_gt = smpl_gt_params[:, 82:85]

		smpl_model = smpl_m if gt[:, 158].item() > 0.5 else smpl_f

		out_path = os.path.join(output_dir, f"{label}_{idx:03d}.png")
		render_smpl_comparison(
			smpl_model, faces,
			body_pose_pred, global_orient_pred, betas_pred, transl_pred, joints_pred.squeeze().detach().cpu().numpy(),
			pose_gt, global_gt, betas_gt, transl_gt, joints_gt.squeeze().detach().cpu().numpy(),
			out_path
		)

	print(f"\n✅ Saved images for: {selected_df[['label', 'idx']].to_string(index=False)}")

if __name__ == "__main__":
	main()
