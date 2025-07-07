# render_utils.py

import numpy as np
import pyrender
import trimesh
from PIL import Image

def render_smpl_comparison(model, faces,
						   body_pose_1, global_orient_1, betas_1, transl_1, joints_1,
						   body_pose_2, global_orient_2, betas_2, transl_2, joints_2,
						   out_path, joint_radius=0.015, ground_size=3.0):

	scene = pyrender.Scene()
	mesh_color_1 = (0.2, 0.6, 1.0, 1.0)    # pred
	mesh_color_2 = (0.6, 0.2, 1.0, 1.0)    # gt
	joints_color_1 = (0.9, 0.3, 0.2, 1.0)  # pred
	joints_color_2 = (0.2, 0.9, 0.3, 1.0)  # gt

	smpl_1 = model(body_pose=body_pose_1, global_orient=global_orient_1, betas=betas_1, transl=transl_1)
	smpl_2 = model(body_pose=body_pose_2, global_orient=global_orient_2, betas=betas_2, transl=transl_2)

	mesh1 = pyrender.Mesh.from_trimesh(trimesh.Trimesh(smpl_1.vertices.squeeze().detach().cpu().numpy(), faces, process=False),
										material=pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh_color_1))
	mesh2 = pyrender.Mesh.from_trimesh(trimesh.Trimesh(smpl_2.vertices.squeeze().detach().cpu().numpy(), faces, process=False),
									   material=pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh_color_2))

	scene.add(mesh1)
	scene.add(mesh2)

	for joint in joints_1:
		marker = trimesh.creation.icosphere(radius=joint_radius)
		marker.apply_translation(joint)
		scene.add(pyrender.Mesh.from_trimesh(marker,
											 material=pyrender.MetallicRoughnessMaterial(baseColorFactor=joints_color_1),
											 smooth=False))

	for joint in joints_2:
		marker = trimesh.creation.icosphere(radius=joint_radius)
		marker.apply_translation(joint)
		scene.add(pyrender.Mesh.from_trimesh(marker,
											 material=pyrender.MetallicRoughnessMaterial(baseColorFactor=joints_color_2),
											 smooth=False))

	ground = trimesh.creation.box(extents=[ground_size, ground_size, 0.01])
	ground.apply_translation([0, 0, -0.005])
	scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))

	cam_pose = np.array([[1.0, 0.0, 0.0, 1.0],
						 [0.0, 1.0, 0.0, 1.25],
						 [0.0, 0.0, 1.0, 2.0],
						 [0.0, 0.0, 0.0, 1.0]])
	scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
	scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=cam_pose)

	renderer = pyrender.OffscreenRenderer(800, 600)
	color, _ = renderer.render(scene)
	Image.fromarray(color).save(out_path)
	renderer.delete()
