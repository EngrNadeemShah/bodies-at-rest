import torch

def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""
	Convert axis-angle vectors to rotation matrices via Rodrigues’ formula.
	axis_angle: (..., 3)
	Returns R: (..., 3, 3)
	"""
	# angle = ||v||
	theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(eps)      # (..., 1)
	axis  = axis_angle / theta                                         # (..., 3)

	# build skew-symmetric cross-product matrix K
	K = torch.zeros((*axis.shape[:-1], 3, 3), device=axis.device, dtype=axis.dtype)
	K[..., 0, 1] = -axis[..., 2]
	K[..., 0, 2] =  axis[..., 1]
	K[..., 1, 0] =  axis[..., 2]
	K[..., 1, 2] = -axis[..., 0]
	K[..., 2, 0] = -axis[..., 1]
	K[..., 2, 1] =  axis[..., 0]

	I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)  # (...,3,3)
	sin_t = torch.sin(theta)[..., None]                                  # (...,1,1)
	cos_t = torch.cos(theta)[..., None]

	# Rodrigues formula: R = I + sinθ K + (1–cosθ) K²
	R = I + sin_t * K + (1 - cos_t) * (K @ K)
	return R
