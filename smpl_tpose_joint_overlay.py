import numpy as np
import torch
import pyrender
import trimesh
import matplotlib.pyplot as plt
from smplx import SMPL
import os

# ------------------ CONFIG ------------------
smpl_male_path = "/home/nadeemshah/coding/bodies-at-rest/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]
PROXIMAL_JOINTS = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]
DISTAL_JOINTS = [i for i in range(24) if i not in PROXIMAL_JOINTS]

SKELETON = [
    (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15), (3, 13), (3, 14),
    (13, 16), (16, 18), (18, 20), (20, 22),
    (14, 17), (17, 19), (19, 21), (21, 23)
]

# ------------------ SMPL ------------------
model = SMPL(smpl_male_path).to("cpu")
betas = torch.zeros((1, 10))
global_orient = torch.zeros((1, 3))
body_pose = torch.zeros((1, 69))
transl = torch.zeros((1, 3))

output = model(betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl, return_verts=True)

# After output = model(...), flip model + joints vertically
joints = output.joints[0, :24].detach().cpu().numpy()
verts = output.vertices[0].detach().cpu().numpy()

# Flip everything upright in camera space (rotate 180° around X-axis)
flip_matrix = np.diag([1, -1, -1])
rotate_y = np.array([[-1, 0,  0],
                     [ 0, 1,  0],
                     [ 0, 0, -1]])
joints = joints @ rotate_y
verts = verts @ rotate_y
joints = joints @ flip_matrix
verts = verts @ flip_matrix

faces = model.faces

# ------------------ SCENE & CAMERA ------------------
scene = pyrender.Scene()
mesh_color = (0.7, 0.7, 0.7, 0.25)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
scene.add(pyrender.Mesh.from_trimesh(mesh,
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh_color)))

width, height = 800, 800
fov_y = np.pi / 3.0
f = 0.5 * height / np.tan(fov_y / 2.0)
K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

cam_pose = np.eye(4)
cam_pose[:3, 3] = [0, 0, 2.5]
camera = pyrender.PerspectiveCamera(yfov=fov_y)
scene.add(camera, pose=cam_pose)
scene.add(pyrender.DirectionalLight(np.ones(3), intensity=3.0), pose=cam_pose)

# ------------------ RENDER ------------------
r = pyrender.OffscreenRenderer(width, height)
color, _ = r.render(scene)
r.delete()

# ------------------ PROJECT JOINTS ------------------
joints_h = np.hstack([joints, np.ones((joints.shape[0], 1))])
joints_cam = (cam_pose @ joints_h.T).T[:, :3]
joints_2d = joints_cam[:, :2] / joints_cam[:, 2:3]
joints_2d = (K[:2, :2] @ joints_2d.T).T + K[:2, 2]

# ------------------ DRAW SKELETON ------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.flipud(color))  # flip image vertically so head is at top

# Skeleton lines
for i, j in SKELETON:
    x = [joints_2d[i, 0], joints_2d[j, 0]]
    y = [joints_2d[i, 1], joints_2d[j, 1]]
    ax.plot(x, y, 'k-', linewidth=2, alpha=0.6)

# Smart label placement
offsets = {
    "left": (-60, 5),
    "right": (20, 5),
    "above": (0, -12),
    "below": (0, 12),
    "left_above": (-60, -12),
    "right_above": (20, -12),
    "left_below": (-60, 20),
    "right_below": (20, 20)
}
# Place labels based on joint index (customized for clarity)
placement = {
    0: "below",     # Pelvis
    1: "left", 2: "right",        # Hips
    3: "below", 6: "below", 9: "below", 12: "above", 15: "above",  # Spine to head
    4: "left", 5: "right",        # Knees
    7: "left", 8: "right",        # Ankles
    10: "left_below", 11: "right_below",      # Feet
    13: "left_above", 14: "right_above",  # Collars
    16: "left_above", 17: "right_above",  # Shoulders
    18: "left", 19: "right",              # Elbows
    20: "left", 21: "right",              # Wrists
    22: "left_below", 23: "right_below"   # Hands
}

# Joint markers
for i, (x, y) in enumerate(joints_2d):
    col = 'red' if i in PROXIMAL_JOINTS else 'blue'
    dx, dy = offsets.get(placement.get(i, "above"))
    ax.scatter(x, y, c=col, s=40)
    ax.text(x + dx, y + dy, f"{JOINT_NAMES[i]}({i+1})", fontsize=6, color=col, weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

ax.axis("off")
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/smpl_tpose_joint_overlay.png", dpi=300)
plt.show()
