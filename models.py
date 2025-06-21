import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# When loss_type == 'anglesDC'
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


class PressureNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=88, use_relu=False):
		super(PressureNet, self).__init__()

		# Move bounds to the appropriate device
		self.bounds = bounds.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),	# (3, 128, 54) -> (192, 64, 27)
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),					# (192, 64, 27)
			nn.Dropout(p=0.1),
			nn.MaxPool2d(3, stride=2),											# (192, 64, 27) -> (192, 31, 13)
			nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),			# (192, 31, 13) -> (192, 29, 11)
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),					# (192, 29, 11)
			nn.Dropout(p=0.1),
			nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),			# (192, 29, 11) -> (384, 27, 9)
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),					# (384, 27, 9)
			nn.Dropout(p=0.1),
			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),			# (384, 27, 9) -> (384, 25, 7)
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),					# (384, 25, 7)
			nn.Dropout(p=0.1),
		)

		# self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Reduce feature map size

		self.output_layer = nn.Sequential(
			nn.Linear(67200, num_classes)
		)

	def forward(self, x):
		# Feature extraction
		x = self.features(x)		# (B, in_channels, 128, 54) -> (B, 384, 25, 7)
		x = torch.flatten(x, 1)		# (B, 384, 25, 7) -> (B, 384*25*7=67200)
		x = self.output_layer(x)	# (B, 67200) -> (B, 88)

		# # 1. Scale adjustment
		# x *= 0.01		# todo Apply selective scaling (0.01 only on translation, not entire vector)

		# # 2. Pad to match SMPL format: from 88 → 91 dims
		# # Extra 3 slots are reserved for copying and transforming Cartesian root to angle
		# x = F.pad(x, (0, 3))	# (B, 88) -> (B, 91)

		# # 3. Shift body_pose (23 joints * 3D axis-angle = 69 dims)
		# # to the end from x[:, 19:88] to x[:, 22:91]
		# body_pose = x[:, 19:88].clone()
		# x[:, 22:91] = body_pose

		# # 4. Normalize betas (shape parameters) to be within ~[-3, 3]
		# x[:, 0:10] = torch.tanh(x[:, 0:10] / 3) * 3

		# # 5. Apply fixed offset to root translation (empirically derived)
		# x[:, 10:13] += torch.tensor([0.6, 1.2, 0.1], device=x.device)
		# # E.g., if subject is lying on a bed → values like [0.0, 1.0, 0.8]

		# # 6. Safe atan2 conversion of root rotation from 6D (sin, cos) components to axis-angle representation
		# def safe_atan2(y, x, eps=1e-6):
		# 	return torch.atan2(y + eps * (y == 0).float(), x + eps * (x == 0).float())

		# root_sin = x[:, [16, 17, 18]]
		# root_cos = x[:, [13, 14, 15]]
		# x[:, 19:22] = safe_atan2(root_sin, root_cos)

		# 7. Apply bounds-based tanh normalization on body_pose
		# Calculate bounds mean and difference
		bounds_mean = self.bounds.mean(dim=1)	# (72, 2) -> (72,)
		bounds_diff = self.bounds[:, 1] - self.bounds[:, 0]	# (72,)

		# Normalize using bounds mean and diff
		body_pose_normalized = (x[:, 19:91] - bounds_mean) * (2.0 / bounds_diff)

		# Clip to [-1, 1] range
		body_pose_clipped = torch.tanh(body_pose_normalized)

		# Rescale back to original bounds
		x[:, 19:91] = body_pose_clipped / (2.0 / bounds_diff) + bounds_mean

		return x