import numpy as np
import torch
import torch.nn.functional as F
from utils import apply_global_rigid_transformations, convert_axis_angle_to_rotation_matrix

class SMPLPreloader:
	def __init__(self, smpl_male, smpl_feml, device):
		self.device = device
		self.smpl_male = smpl_male
		self.smpl_feml = smpl_feml

	def smpl_forward(self, predicted_labels, true_labels):
		"""
		Dynamically runs the appropriate SMPL model based on gender.
		Returns predicted joints from SMPL given model outputs.
		"""
		betas = predicted_labels[:, 0:10]
		transl = predicted_labels[:, 10:13]
		global_orient = predicted_labels[:, 19:22].unsqueeze(1)  # (B, 1, 3)
		body_pose = predicted_labels[:, 22:91].view(-1, 23, 3)   # (B, 23, 3)

		gender_flags = true_labels[:, 157:159]  # g1 (female), g2 (male)
		is_male = gender_flags[:, 1].bool()
		is_female = gender_flags[:, 0].bool()

		predicted_joint_positions = torch.zeros(predicted_labels.size(0), 24, 3, device=self.device)

		if is_male.any():
			smpl_output_m = self.smpl_male(
				betas=betas[is_male],
				body_pose=body_pose[is_male],
				global_orient=global_orient[is_male],
				transl=transl[is_male]
			)
			predicted_joint_positions[is_male] = smpl_output_m.joints[:, :24]

		if is_female.any():
			smpl_output_f = self.smpl_feml(
				betas=betas[is_female],
				body_pose=body_pose[is_female],
				global_orient=global_orient[is_female],
				transl=transl[is_female]
			)
			predicted_joint_positions[is_female] = smpl_output_f.joints[:, :24]

		return predicted_joint_positions
