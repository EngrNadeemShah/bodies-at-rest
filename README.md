# Supplementary Code for Master's Thesis

**Thesis Title:**  
*3D Human Pose and Shape Estimation from Pressure Images Using Synthetic Data*

**Author:**  
Nadeem Shah  
Otto von Guericke University Magdeburg

---

## 📂 Contents

This folder contains curated Python code files directly related to the experiments and results presented in the thesis. It includes model training, evaluation, and visualization scripts.

### 🔧 Core Scripts

- `train.py` – Trains the deep learning model on the `straight_limbs` subset of the PressurePose dataset.
- `evaluate_best_model.py` – Evaluates the trained model on the held-out test set and computes MPJPE and v2v errors.
- `plot_evaluation_results.py` – Generates plots of MPJPE/v2v distributions, jointwise errors, and gender-specific breakdowns.
- `generate_visualizations.py` – Renders SMPL mesh predictions vs. ground truth for representative test samples.

### 🧹 Preprocessing

- `preprocess_data_straight_limbs.py` – Converts and prepares synthetic pressure data (from `.p` to `.hdf5`) with normalization.

### 🧠 Model and Utilities

- `models_multi_head.py` – Defines the CNN model architecture with modular heads for SMPL parameter regression.
- `smpl_class.py` – Wrapper for SMPL model integration to generate joints and mesh vertices.
- `render_utils.py` – Functions to render SMPL meshes and overlay joints for visualization.
- `utils.py` – General-purpose utilities (e.g., loading, formatting, logging).
- `utils_geometry.py` – Functions for rotation matrix operations, geodesic loss, and axis-angle conversions.
- `datasets.py` – HDF5-based PyTorch `Dataset` loader with input normalization support.

### 📁 Other

- `requirements.yaml` – Conda environment specification to reproduce the experiments.
- `stats/` – Contains global statistics (mean, std) used for input normalization.
- `runs/` – (Optional) Directory where TensorBoard logs or intermediate checkpoints may be saved.

---

## 📁 Dataset

This code expects preprocessed `.hdf5` files derived from the `straight_limbs` subset of the PressurePose dataset. These are not included in this archive due to size constraints.

---

## 📜 License

This code is shared exclusively as supplementary material for academic evaluation. For any reuse or adaptation, please contact the author.

