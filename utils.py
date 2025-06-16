import torch
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import trimesh
import pyrender

import matplotlib as mpl
mpl.use('TkAgg')


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

def retrieve_data_file_paths(folder, verbose=False):
	total_files = 0
	file_paths = []
	for root, dirs, files in os.walk(folder):
		print('Root:', root) if verbose else None
		for file_index, file in enumerate(files):
			if file.endswith('.p'):
				file_path = os.path.join(root, file)
				file_paths.append(file_path)
				total_files += 1
				print(f'{file_index+1:02d} ({total_files:02d}): {file}') if verbose else None
	return file_paths

def plot_input_channels(inputs_batch, batch_idx=0):
	num_channels = inputs_batch.shape[1]
	num_cols = min(num_channels, 5)
	num_rows = (num_channels + num_cols - 1) // num_cols
	fig, axes = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))
	axes = axes.flatten() if num_rows > 1 else axes

	for i in range(num_channels):
		axes[i].imshow(inputs_batch[0, i].cpu().numpy())
		axes[i].set_title(f'Input Channel {i}\n{inputs_batch[0, i].shape} | {inputs_batch[0, i].dtype}\nmin: {inputs_batch[0, i].min():.2f} | max: {inputs_batch[0, i].max():.2f}\nmean: {inputs_batch[0, i].mean():.2f} | std: {inputs_batch[0, i].std():.2f}', fontsize=8, pad=10)
		axes[i].axis('off')

	for j in range(i + 1, len(axes)):
		axes[j].axis('off')

	fig.suptitle(f'Batch Index: {batch_idx + 1}', fontsize=16)
	plt.tight_layout()
	plt.show()

def log_tensor_mean_std_by_dim(name, tensor, dims=(0,), limit=24):
	"""
	Logs per-dimension mean and std in the format: mean (std)

	Args:
	- name: Name of the tensor
	- tensor: PyTorch tensor
	- dims: Dimensions to reduce over (e.g., batch)
	- limit: Max number of elements to display
	"""
	if not isinstance(tensor, torch.Tensor):
		print(f"{name}: [Not a tensor]")
		return

	shape = tuple(tensor.shape)
	print(f"\n🔹 {name} — shape: {shape}, reduced over dims={dims}")

	# Compute mean and std over dims
	mean = tensor.mean(dim=dims)
	std = tensor.std(dim=dims)

	# If it's 1D (like C,)
	if mean.ndim == 1:
		mean_np = mean.cpu().numpy()
		std_np = std.cpu().numpy()
		for i in range(min(len(mean_np), limit)):
			print(f"   [{i:02d}] {mean_np[i]:+.4f} ({std_np[i]:.4f})")
		if len(mean_np) > limit:
			print("   ...")

	# If it's 2D (like joints × coords)
	elif mean.ndim == 2:
		mean_np = mean.cpu().numpy()
		std_np = std.cpu().numpy()
		for i in range(min(mean_np.shape[0], limit)):
			line = "   [{:02d}] ".format(i)
			line += ", ".join(f"{m:+.4f} ({s:.4f})" for m, s in zip(mean_np[i], std_np[i]))
			print(line)

	else:
		print("   [Too high-dimensional to display]")

def format_stats(tensor, name):
	min_val = tensor.min().item()
	max_val = tensor.max().item()
	mean_val = tensor.mean().item()
	std_val = tensor.std().item()
	def fmt(x):
	# Format: always show sign, (if :+010.2f)pad to width 8, 2 decimals, leading zeros
	# Example: +002500.42, -000001.65
		return f"{x:+06.2f}"
	print(f"{name} -> min: {fmt(min_val)}, max: {fmt(max_val)}, mean: {fmt(mean_val)}, std: {fmt(std_val)}")

def print_mean_of_model_weights_and_gradients(model, message="Model Parameters and Gradients (Mean)"):
	"""Prints the mean of model parameters and their gradients."""
	print(f"\n--- {message} ---")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(f"{name}:\tmean={param.data.mean():.6f}, grad={param.grad.mean().item() if param.grad is not None else 'None'}")

def visualize_smpl_with_joints(model,
                               body_pose_1, global_orient_1, betas_1, transl_1,
                               joints_1=None,
                               body_pose_2=None, global_orient_2=None, betas_2=None, transl_2=None,
                               joints_2=None,
                               show_ground=True, joint_radius=0.015, ground_size=3.0):

    mesh_color_1 = (0.2, 0.6, 1.0, 1.0)	# blue
    mesh_color_2 = (0.6, 0.2, 1.0, 1.0)	# purple
    faces = model.faces
    scene = pyrender.Scene()

    # --- SMPL Model 1
    smpl_output_1 = model(body_pose=body_pose_1, global_orient=global_orient_1, betas=betas_1, transl=transl_1)
    vertices_1 = smpl_output_1.vertices.detach().cpu().numpy().squeeze()
    mesh_1 = trimesh.Trimesh(vertices_1, faces, process=False)
    mesh_1 = pyrender.Mesh.from_trimesh(mesh_1, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh_color_1))
    scene.add(mesh_1)

    # --- Optional Joint Markers (Set 1)
    if joints_1 is not None:
        for joint in joints_1:
            sphere = trimesh.creation.icosphere(radius=joint_radius)
            sphere.apply_translation(joint)
            marker_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            scene.add(marker_mesh)

    # --- SMPL Model 2
    if body_pose_2 is not None and global_orient_2 is not None and betas_2 is not None and transl_2 is not None:
        smpl_output_2 = model(body_pose=body_pose_2, global_orient=global_orient_2, betas=betas_2, transl=transl_2)
        vertices_2 = smpl_output_2.vertices.detach().cpu().numpy().squeeze()
        mesh_2 = trimesh.Trimesh(vertices_2, faces, process=False)
        mesh_2 = pyrender.Mesh.from_trimesh(mesh_2, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh_color_2))
        scene.add(mesh_2)

        # --- Optional Joint Markers (Set 2)
        if joints_2 is not None:
            for joint in joints_2:
                sphere = trimesh.creation.icosphere(radius=joint_radius)
                sphere.apply_translation(joint)
                marker_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
                scene.add(marker_mesh)

    # --- Ground plane
    if show_ground:
        ground = trimesh.creation.box(extents=[ground_size, ground_size, 0.01])
        ground.apply_translation([0, 0, -0.005])
        scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))

    # --- Show scene
    pyrender.Viewer(scene, use_raymond_lighting=True)

def plot_single_channel(input_image, batch_idx=0, title='Image'):
	if isinstance(input_image, torch.Tensor):
		input_image = input_image.cpu().numpy()

	plt.imshow(input_image)
	plt.title(f'{title} | {input_image.shape} | {input_image.dtype}\nmin: {input_image.min():.2f} | max: {input_image.max():.2f}\nmean: {input_image.mean():.2f} | std: {input_image.std():.2f}', fontsize=8, pad=10)
	plt.axis('off')
	plt.show()
