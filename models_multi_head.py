import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# These are the min-max bounds for the 72 axis-angle rotation vectors (3 for root orientation, 69 for body pose).
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
	[-0.01, 0.01],
	[-0.01, 0.01],
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
	[-0.01, 0.01],
	[-0.01, 0.01],
	[-0.01, 0.01],    # hand
	[-0.01, 0.01],
	[-0.01, 0.01],
	[-0.01, 0.01]
]) * 1.2


class PressureNetMultiHead(nn.Module):
	def __init__(self, in_channels=3, use_relu=True):
		super().__init__()

		# Shared CNN feature extractor
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),	# (3, 128, 54) -> (192, 64, 27)
			nn.BatchNorm2d(192),
			nn.ReLU() if use_relu else nn.Tanh(),
			nn.Dropout(0.1),

			nn.MaxPool2d(3, stride=2),											# (192, 64, 27) -> (192, 31, 13)

			nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),			# (192, 31, 13) -> (192, 29, 11)
			nn.BatchNorm2d(192),
			nn.ReLU() if use_relu else nn.Tanh(),
			nn.Dropout(0.1),

			nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),			# (192, 29, 11) -> (384, 27, 9)
			nn.BatchNorm2d(384),
			nn.ReLU() if use_relu else nn.Tanh(),
			nn.Dropout(0.1),

			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),			# (384, 27, 9) -> (384, 25, 7)
			nn.BatchNorm2d(384),
			nn.ReLU() if use_relu else nn.Tanh(),
			nn.Dropout(0.1),
		)

		# Global adaptive pooling to reduce spatial dimensions
		self.adapt_pool = nn.AdaptiveAvgPool2d(1)		# (B, 384, 25, 7) → (B, 384, 1, 1)

		# Shared MLP bottleneck
		self.fc_shared = nn.Sequential(
			nn.Linear(384, 2048),						# (B, 384) → (B, 2048)
			nn.ReLU() if use_relu else nn.Tanh(),
			nn.Dropout(0.1),
		)

		# Separate heads
		self.head_betas = nn.Linear(2048, 10)
		self.head_global_orient_6d = nn.Linear(2048, 6)	# 6D root rotation (3 axes, each as sin+cos pair)
		self.head_body_pose = nn.Linear(2048, 69)		# remaining 23 joints rotation_vectors
		self.head_transl = nn.Linear(2048, 3)

		# Learnable translation bias, and store bounds as a buffer
		self.transl_bias = nn.Parameter(torch.tensor([0.6, 1.2, 0.1]))
		self.register_buffer("bounds", bounds)

	def _clip_into_bounds(self, raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
		bm = bounds.mean(dim=1)
		bd = (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-6)
		norm = (raw - bm) * (2.0 / bd)
		clip = torch.tanh(norm)
		return clip / (2.0 / bd) + bm

	def forward(self, x):								# (B, 3, 128, 54)
		# shared features
		x = self.features(x)							# (B, 384, 25, 7)
		x = self.adapt_pool(x).view(x.shape[0], -1)		# → (B, 384)
		x = self.fc_shared(x)							# → (B, 2048)

		# 1) Betas → soft‐clip into [–3, 3]
		betas = torch.tanh(self.head_betas(x) / 3) * 3	# (B, 10)

		# 2) Root rotation: 6D → normalized sin/cos → atan2 → clip into bounds[:3]
		global_orient_6d = self.head_global_orient_6d(x).view(-1, 3, 2)	# (B, 3, 2)
		global_orient_6d = F.normalize(global_orient_6d, dim=2)
		sin, cos = global_orient_6d[:, :, 1], global_orient_6d[:, :, 0]
		global_orient_aa_raw = torch.atan2(sin, cos)	# (B, 3)
		root_bounds = self.bounds[:3]				# (3, 2)
		global_orient_aa = self._clip_into_bounds(global_orient_aa_raw, root_bounds)

		# 3) Body pose → raw → clip into bounds[3:]
		body_pose_aa_raw = self.head_body_pose(x)		# (B, 69)
		body_bounds = self.bounds[3:]				# (69, 2)
		body_pose_aa = self._clip_into_bounds(body_pose_aa_raw, body_bounds)

		# 4) Translation → scale to meters + bias
		transl = self.head_transl(x) + self.transl_bias  # (B, 3)		# Removed scaling (* 0.01)

		# 5) Final SMPL vector = 10+3+3+69 = 85 dims
		out = torch.cat([betas, global_orient_aa, body_pose_aa, transl], dim=1)
		return out