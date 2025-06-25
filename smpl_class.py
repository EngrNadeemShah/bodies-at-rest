import torch

class SMPLPreloader:
	def __init__(self, smpl_male, smpl_feml, device):
		self.device = device
		self.smpl_male = smpl_male
		self.smpl_feml = smpl_feml

	def smpl_forward(self, betas, global_orient, body_pose, transl, ground_truth):
		"""
		Dynamically runs the appropriate SMPL model based on gender.
		Returns predicted joints from SMPL given model outputs.
		"""

		global_orient = global_orient.unsqueeze(1)	# (B, 3) -> (B, 1, 3)

		# Fetch gender flags from ground truth
		gender_flags = ground_truth[:, 157:159]		# g1 (female), g2 (male)
		is_male = gender_flags[:, 1].bool()
		is_female = gender_flags[:, 0].bool()

		joints_pred = torch.zeros(betas.size(0), 24, 3, device=self.device)

		if is_male.any():
			smpl_output_m = self.smpl_male(
				betas=betas[is_male],
				body_pose=body_pose[is_male],
				global_orient=global_orient[is_male],
				transl=transl[is_male]
			)
			joints_pred[is_male] = smpl_output_m.joints[:, :24]

		if is_female.any():
			smpl_output_f = self.smpl_feml(
				betas=betas[is_female],
				body_pose=body_pose[is_female],
				global_orient=global_orient[is_female],
				transl=transl[is_female]
			)
			joints_pred[is_female] = smpl_output_f.joints[:, :24]

		return joints_pred
