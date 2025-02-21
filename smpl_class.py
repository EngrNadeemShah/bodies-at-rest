import numpy as np
import torch


class SMPLPreloader:
	def __init__(self, smpl_male, smpl_feml, device, config, vertices):

		self.device = device
		self.batch_size = config['batch_size']
		self.vertices = vertices


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

	def get_parameters(self, gender_labels):
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

		# Exclude the last batch if it is not full size, as parameters are expanded to full batch size.

		# Reshape parameters to match the previous shapes
		shapedirs = shapedirs.view(self.batch_size, 10, -1)		# (B, 10, 6890*3)
		J_regressor = J_regressor.permute(0, 2, 1)				# (B, 6890, 24)
		posedirs = posedirs.view(self.batch_size, 207, -1)		# (B, 207, 10*3)

		return shapedirs, v_template, J_regressor, posedirs, weights, BRD_shape