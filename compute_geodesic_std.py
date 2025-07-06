import h5py
import numpy as np
from utils import get_preprocessed_hdf5_path

# 1. Point this to your pre-processed HDF5:
hdf5_file_name = 'preprocessed_straight_limbs_mod1_add_noise_0__include_weight_height_False__omit_contact_sobel_False__use_hover_False__mod_1__normalize_per_image_True.hdf5'
hdf5_path = get_preprocessed_hdf5_path(hdf5_file_name)

# 2. Accumulate all axis-angle labels in “train”:
all_labels = []
with h5py.File(hdf5_path, 'r') as f:
    train_grp = f['train']
    for pose in train_grp:                   # e.g. 'straight_limbs'
        for gender in train_grp[pose]:       # e.g. 'f', 'm'
            lbl = train_grp[pose][gender]['labels'][:]  # shape (N,162)
            all_labels.append(lbl)
all_labels = np.concatenate(all_labels, axis=0)      # shape (N_total, 162)

print(f"all_labels.shape: {all_labels.shape}")

# 3. Slice out the two blocks of axis-angle dims:
global_aa = all_labels[:,  82:  85]   # root joint (B,3)
body_aa   = all_labels[:,  85: 154]   # 23 joints flattened as (B,69)

print(f"global_aa.shape: {global_aa.shape}")
print(f"body_aa.shape: {body_aa.shape}")

# 4. Compute per-sample geodesic (i.e. ‖axis-angle‖₂) and take std:
global_angles = np.linalg.norm(global_aa, axis=1)       # (N_total,)
body_angles   = np.linalg.norm(body_aa.reshape(-1,3), axis=1)  # (N_total*23,)

print(f"global_angles.shape: {global_angles.shape}")
print(f"body_angles.shape: {body_angles.shape}")

geo_std_global = global_angles.std()
geo_std_body   = body_angles.std()

print(f"geo_std_global = {geo_std_global:.6f} rad")
print(f"geo_std_body   = {geo_std_body:.6f} rad")

geo_mean_global = global_angles.mean()
geo_mean_body   = body_angles.mean()

print(f"geo_mean_global = {geo_mean_global:.6f} rad")
print(f"geo_mean_body   = {geo_mean_body:.6f} rad")