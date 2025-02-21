import torch
import numpy as np
import torch.nn.functional as F


def convert_axis_angle_to_rotation_matrix(theta):
	"""
	Convert axis-angle representation to rotation matrix using Rodrigues' rotation formula.

	Args:
		theta (torch.Tensor): Tensor of shape (N, 72) representing the 24 axis-angle vectors.

	Returns:
		torch.Tensor: Tensor of shape (N, 3, 3) representing the rotation matrices.
	"""

	# Calculate the norm of theta
	l1norm = torch.norm(theta + 1e-8, p=2, dim=2)
	angle = torch.unsqueeze(l1norm, -1)

	# Normalize theta
	normalized = torch.div(theta, angle)
	angle = angle * 0.5

	# Compute quaternion
	v_cos = torch.cos(angle)
	v_sin = torch.sin(angle)
	quat = torch.cat([v_cos, v_sin * normalized], dim=2)

	# Normalize quaternion
	norm_quat = quat / quat.norm(p=2, dim=2, keepdim=True)

	# Extract quaternion components
	w, x, y, z = norm_quat[:, :, 0], norm_quat[:, :, 1], norm_quat[:, :, 2], norm_quat[:, :, 3]
	w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
	wx, wy, wz = w * x, w * y, w * z
	xy, xz, yz = x * y, x * z, y * z

	# Compute rotation matrix
	rotMat = torch.stack([
		w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
		2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
		2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
	], dim=2)

	rotMat = rotMat.view(-1, 24, 3, 3)

	return rotMat

def apply_global_rigid_transformations(rotation_matrices, joint_locations, parents, device, rotate_base=False):
	"""
	Perform batch global rigid transformation.

	Args:
		rotation_matrices (torch.Tensor): Rotation matrices of shape (N, 24, 3, 3).
		joint_locations (torch.Tensor): Joint locations of shape (N, 24, 3).
		parents (list): List of parent indices.
		device (torch.device): Device to perform computation.
		rotate_base (bool): Flag to rotate base.

	Returns:
		tuple: A tuple containing:
			- torch.Tensor: Transformed joint locations of shape (N, 24, 3).
			- torch.Tensor: Transformation matrices of shape (N, 24, 4, 4).
	"""

	batch_size = rotation_matrices.shape[0]

	if rotate_base:
		np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)	# (3, 3)
		rot_x = torch.tensor(np_rot_x, dtype=torch.float32, device=device).repeat(batch_size, 1, 1)	# (N, 3, 3)
		root_rotation = torch.matmul(rotation_matrices[:, 0], rot_x)	# (N, 3, 3)
	else:
		root_rotation = rotation_matrices[:, 0]		# (N, 3, 3)

	joint_locations = joint_locations.unsqueeze(-1)	# (N, 24, 3, 1)

	def make_transformation_matrix(rotation, translation):	# (N, 3, 3), (N, 3, 1)
		rotation_homo = F.pad(rotation, (0, 0, 0, 1))		# (0+N+0, 0+3+1, 0+3+0) -> (N, 4, 3)
		translation_homo = torch.cat([translation, torch.ones(batch_size, 1, 1, device=device)], dim=1)		# (N, 4, 1)
		return torch.cat([rotation_homo, translation_homo], dim=2)		# (N, 4, 4)

	root_transformation = make_transformation_matrix(root_rotation, joint_locations[:, 0])	# (N, 4, 4)
	transformations = [root_transformation]	# [(N, 4, 4)]

	for i in range(1, len(parents)):	# (1:24) -> 23 iterations
		relative_translation = joint_locations[:, i] - joint_locations[:, parents[i]]	# (N, 3, 1)
		current_transformation = make_transformation_matrix(rotation_matrices[:, i], relative_translation)	# (N, 4, 4)
		parent_transformation = transformations[parents[i]]	# (N, 4, 4)
		full_transformation = torch.matmul(parent_transformation, current_transformation)	# (N, 4, 4)
		transformations.append(full_transformation)	# [24x (N, 4, 4)]

	transformations = torch.stack(transformations, dim=1)		# (N, 24, 4, 4)
	transformed_joint_locations = transformations[:, :, :3, 3]	# (N, 24, 3)

	joint_locations_homo = torch.cat([joint_locations, torch.zeros(batch_size, 24, 1, 1, device=device)], dim=2)	# (N, 24, 4, 1)
	initial_bone = torch.matmul(transformations, joint_locations_homo)	# (N, 24, 4, 4) x (N, 24, 4, 1) -> (N, 24, 4, 1)
	initial_bone = F.pad(initial_bone, (3, 0))		# (N, 24, 4, 3+1+0) -> (N, 24, 4, 4)
	transformation_matrices = transformations - initial_bone	# (N, 24, 4, 4)

	return transformed_joint_locations, transformation_matrices

def print_error_summary(true_markers_xyz, predicted_markers_xyz, verbose=True):
	true_markers_xyz = true_markers_xyz.view(-1, 24, 3)
	predicted_markers_xyz = predicted_markers_xyz.view(-1, 24, 3)

	error = predicted_markers_xyz - true_markers_xyz	# (N, 24, 3)
	error_norm = torch.norm(error, dim=2, keepdim=True)	# (N, 24, 1)
	error = torch.cat((error, error_norm), dim=2)		# (N, 24, 4)

	error_avg = error.mean(dim=0) / 10		# convert from mm to cm		# (24, 4)
	error_avg_print = error_avg.cpu().numpy()	# (24, 4)

	joint_names = [
		'Pelvis', 'L Hip', 'R Hip', 'Spine 1', 'L Knee', 'R Knee',
		'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot', 'R Foot',
		'Neck', 'L Sh.in', 'R Sh.in', 'Head', 'L Sh.ou', 'R Sh.ou',
		'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand', 'R Hand'
	]

	if verbose:
		print(f"{'Joint':<10} {'x, cm':<10} {'y, cm':<10} {'z, cm':<10} {'norm':<10}")
		for i, joint in enumerate(joint_names):
			print(f"{joint:<10} {error_avg_print[i, 0]:<10.2f} {error_avg_print[i, 1]:<10.2f} {error_avg_print[i, 2]:<10.2f} {error_avg_print[i, 3]:<10.2f}")

	error_std = error.std(dim=0) / 10
	error_std_print = error_std.cpu().numpy()	# (24, 4)

	if verbose:
		print("\nStandard Deviation:")
		print(f"{'Joint':<10} {'x, cm':<10} {'y, cm':<10} {'z, cm':<10} {'norm':<10}")
		for i, joint in enumerate(joint_names):
			print(f"{joint:<10} {error_std_print[i, 0]:<10.2f} {error_std_print[i, 1]:<10.2f} {error_std_print[i, 2]:<10.2f} {error_std_print[i, 3]:<10.2f}")

	error_norm = error_norm.squeeze(dim=2)

	return error_norm, error_avg[:, 3], error_std[:, 3]
