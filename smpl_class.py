import numpy as np
import torch
import torch.nn.functional as F
from utils import apply_global_rigid_transformations, convert_axis_angle_to_rotation_matrix


class SMPLPreloader:
	def __init__(self, smpl_male, smpl_feml, device, config):

		self.device = device
		self.batch_size = config['batch_size']
		self.vertices = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
		self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])


		# Load gender-dependent parameters
		self.shapedirs_f	= smpl_feml.shapedirs.permute(2, 0, 1)  # (10, 6890, 3)
		self.shapedirs_m	= smpl_male.shapedirs.permute(2, 0, 1)  # (10, 6890, 3)
		self.v_template_f	= smpl_feml.v_template  # (6890, 3)
		self.v_template_m	= smpl_male.v_template  # (6890, 3)
		self.J_regressor_f	= smpl_feml.J_regressor  # (24, 6890)
		self.J_regressor_m	= smpl_male.J_regressor  # (24, 6890)

		# Select the 10 vertices of interest for posedirs and weights
		if config['mod'] == 1:
			self.posedirs_f	= smpl_feml.posedirs.reshape(207, 6890, 3).index_select(1, torch.tensor(self.vertices, device=self.device))	# (207, 10, 3)
			self.posedirs_m	= smpl_male.posedirs.reshape(207, 6890, 3).index_select(1, torch.tensor(self.vertices, device=self.device))	# (207, 10, 3)
			self.weights_f	= smpl_feml.lbs_weights.index_select(0, torch.tensor(self.vertices, device=self.device))	# (10, 24)
			self.weights_m	= smpl_male.lbs_weights.index_select(0, torch.tensor(self.vertices, device=self.device))	# (10, 24)
		elif config['mod'] == 2:
			self.posedirs_f	= smpl_feml.posedirs.view(-1, 6890, 3)  # (207, 6890, 3)
			self.posedirs_m	= smpl_male.posedirs.view(-1, 6890, 3)  # (207, 6890, 3)
			self.weights_f	= smpl_feml.weights  # (6890, 24)
			self.weights_m	= smpl_male.weights  # (6890, 24)
		else:
			raise ValueError("Invalid configuration for 'mod'. It must be either 1 or 2.")



		# Expand parameters to batch size for efficiency
		self.expand_params()

	def expand_params(self):
		"""Expands all parameters for batch processing."""
		self.shapedirs_f = self.shapedirs_f.unsqueeze(0).expand(self.batch_size, -1, -1, -1)	# (B, 10, 6890, 3)
		self.shapedirs_m = self.shapedirs_m.unsqueeze(0).expand(self.batch_size, -1, -1, -1)	# (B, 10, 6890, 3)
		
		self.v_template_f = self.v_template_f.unsqueeze(0).expand(self.batch_size, -1, -1)		# (B, 6890, 3)
		self.v_template_m = self.v_template_m.unsqueeze(0).expand(self.batch_size, -1, -1)		# (B, 6890, 3)
		
		self.J_regressor_f = self.J_regressor_f.unsqueeze(0).expand(self.batch_size, -1, -1)	# (B, 24, 6890)
		self.J_regressor_m = self.J_regressor_m.unsqueeze(0).expand(self.batch_size, -1, -1)	# (B, 24, 6890)
		
		self.posedirs_f = self.posedirs_f.unsqueeze(0).expand(self.batch_size, -1, -1, -1)		# (B, 207, 10, 3)
		self.posedirs_m = self.posedirs_m.unsqueeze(0).expand(self.batch_size, -1, -1, -1)		# (B, 207, 10, 3)

		self.weights_f = self.weights_f.unsqueeze(0).expand(self.batch_size, -1, -1)			# (B, 10, 24)
		self.weights_m = self.weights_m.unsqueeze(0).expand(self.batch_size, -1, -1)			# (B, 10, 24)

	def fetch_gender_based_parameters(self, gender_labels):
		"""Returns the appropriate SMPL parameters based on gender labels."""
		current_batch_size = gender_labels.shape[0]
		if current_batch_size != self.batch_size:
			self.batch_size = current_batch_size
			self.expand_params()

		gender_idx = gender_labels.long()
		# g2[158] or genders[:, 1]
		#	- 0 for female
		#	- 1 for male

		shapedirs	= torch.stack([self.shapedirs_f,	self.shapedirs_m	], dim=1)[torch.arange(self.batch_size), gender_idx]	# (B, 10, 6890, 3)
		v_template	= torch.stack([self.v_template_f,	self.v_template_m	], dim=1)[torch.arange(self.batch_size), gender_idx]	# (B, 6890, 3)
		J_regressor	= torch.stack([self.J_regressor_f,	self.J_regressor_m	], dim=1)[torch.arange(self.batch_size), gender_idx]	# (B, 24, 6890)
		posedirs	= torch.stack([self.posedirs_f,		self.posedirs_m		], dim=1)[torch.arange(self.batch_size), gender_idx]	# (B, 207, 10, 3)
		weights		= torch.stack([self.weights_f,		self.weights_m		], dim=1)[torch.arange(self.batch_size), gender_idx]	# (B, 10, 24)

		BRD_shape = shapedirs.shape[1:]

		# Reshape parameters to match the previous shapes
		shapedirs = shapedirs.view(self.batch_size, 10, -1)		# (B, 10, 6890*3)
		J_regressor = J_regressor.permute(0, 2, 1)				# (B, 6890, 24)
		posedirs = posedirs.view(self.batch_size, 207, -1)		# (B, 207, 10*3)

		return shapedirs, v_template, J_regressor, posedirs, weights, BRD_shape

	def forward(self, predicted_labels, true_labels):

		batch_size = true_labels.shape[0]

		predicted_label_betas = predicted_labels[:, 0:10].clone()
		predicted_label_root_xyz = predicted_labels[:, 10:13].clone()

		# Convert the body_pose axis-angles to rotation matrices
		predicted_label_angles_rot_mat = convert_axis_angle_to_rotation_matrix(predicted_labels[:, 19:91].view(-1, 24, 3))

		# Extract gender information from labels
		genders = true_labels[:, 157:159].clone()

		# Load SMPL parameters based on gender
		shapedirs, v_template, J_regressor, posedirs, weights, BRD_shape = self.fetch_gender_based_parameters(genders[:, 1])

		# Compute the shaped vertices
		SMPL_pred_v_shaped = (
			torch.bmm(predicted_label_betas.unsqueeze(1), shapedirs)
			.squeeze(1).view(batch_size, BRD_shape[1], BRD_shape[2]) + v_template
		)

		# Extract vertices of interest
		SMPL_pred_v_shaped_red = torch.stack([
			SMPL_pred_v_shaped[:, vertex, :] for vertex in self.vertices
		], dim=1)

		# Compute the pose feature
		predicted_label_angles_rot_mat_pose_feature = (
			(predicted_label_angles_rot_mat[:, 1:] - torch.eye(3, device=self.device))
			.view(-1, 207)
		)

		# Compute the posed vertices
		SMPL_pred_v_posed = (
			torch.bmm(predicted_label_angles_rot_mat_pose_feature.unsqueeze(1), posedirs)
			.view(-1, 10, BRD_shape[2]) + SMPL_pred_v_shaped_red
		)

		# Compute joint locations in 3D
		SMPL_pred_J = (
			torch.matmul(SMPL_pred_v_shaped.transpose(1, 2), J_regressor)
			.permute(0, 2, 1)
		)

		# Apply global rigid transformations to the joint locations
		predicted_label_markers_xyz, SMPL_pred_A = apply_global_rigid_transformations(
			predicted_label_angles_rot_mat, SMPL_pred_J, self.parents, self.device, rotate_base=False
		)

		# Adjust markers by subtracting root joint location
		predicted_label_markers_xyz = (
			predicted_label_markers_xyz - SMPL_pred_J[:, 0:1, :] + predicted_label_root_xyz.unsqueeze(1)
		)

		# Compute final transformation matrices
		SMPL_pred_T = (
			torch.bmm(weights, SMPL_pred_A.view(batch_size, 24, 16))
			.view(batch_size, -1, 4, 4)
		)

		# Compute final vertices
		SMPL_pred_v_posed_homo = torch.cat([
			SMPL_pred_v_posed, torch.ones(batch_size, SMPL_pred_v_posed.shape[1], 1, device=self.device)
		], dim=2)
		
		SMPL_pred_v_homo = torch.matmul(SMPL_pred_T, SMPL_pred_v_posed_homo.unsqueeze(-1))
		SMPL_pred_verts = (
			SMPL_pred_v_homo[:, :, :3, 0] - SMPL_pred_J[:, 0:1, :] + predicted_label_root_xyz.unsqueeze(1)
		)

		# Adjust vertices based on joint addresses
		SMPL_pred_verts_offset = SMPL_pred_verts.clone().detach()
		predicted_label_markers_xyz_detached = predicted_label_markers_xyz.clone().detach()
		synth_joint_addressed = torch.tensor([3, 15, 4, 5, 7, 8, 18, 19, 20, 21], device=self.device)
		SMPL_pred_verts_offset -= predicted_label_markers_xyz_detached[:, synth_joint_addressed, :]

		# Pad predicted labels to increase dimensions
		predicted_labels = F.pad(predicted_labels, (0, 100))

		# Normalize predicted labels using true labels
		predicted_labels[:, 0:10] -= true_labels[:, 72:82]  # Adjust betas
		predicted_labels[:, 10:16] = predicted_labels[:, 13:19].clone()
		predicted_labels[:, 10:13] -= torch.cos(true_labels[:, 82:85].clone())
		predicted_labels[:, 13:16] -= torch.sin(true_labels[:, 82:85].clone())

		# Compute the scaled difference between true and predicted marker positions
		predicted_labels[:, 40:112] = (true_labels[:, :72] / 1000) - predicted_label_markers_xyz.reshape(-1, 72)
		predicted_labels[:, 112:184] = torch.square(predicted_labels[:, 40:112].clone() + 1e-7)
		
		predicted_labels = predicted_labels[:, :40]

		# Normalize predicted labels using standard deviations
		betas_std, body_rot_std, joints_std = 1.728, 0.368, 0.175
		predicted_labels[:, 0:10] /= betas_std
		predicted_labels[:, 10:16] /= body_rot_std
		predicted_labels[:, 16:40] /= joints_std

		return predicted_labels