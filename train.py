import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
from datasets import PressurePoseDataset
import matplotlib.pyplot as plt

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


# # Define the neural network model
# class PressureNet(nn.Module):
# 	def __init__(self, in_channels=1, num_classes=10):
# 		super(PressureNet, self).__init__()

# 		self.features = nn.Sequential(
# 			nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(p=0.1),
# 			nn.MaxPool2d(3, stride=2),
# 			nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(p=0.1),
# 			nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(p=0.1),
# 			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(p=0.1),
# 		)
# 		self.classifier = nn.Sequential(
# 			# nn.Linear(67200, 128),
# 			# nn.ReLU(inplace=True),
# 			# nn.Linear(128, num_classes)
# 			nn.Linear(67200, num_classes)
# 		)

# 	def forward(self, x):
# 		x = self.features(x)
# 		x = torch.flatten(x, 1)
# 		x = self.classifier(x)
# 		return x

# # Training function
# def train(model, device, train_loader, optimizer, criterion, epoch):
# 	model.train()
# 	for batch_idx, (data, target) in enumerate(train_loader):
# 		data, target = data.to(device), target.to(device)
# 		optimizer.zero_grad()
# 		output = model(data)
# 		loss = criterion(output, target)
# 		loss.backward()
# 		optimizer.step()
# 		if batch_idx % 10 == 0:
# 			print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# # Testing function
# def test(model, device, test_loader, criterion):
# 	model.eval()
# 	test_loss = 0
# 	correct = 0
# 	with torch.no_grad():
# 		for data, target in test_loader:
# 			data, target = data.to(device), target.to(device)
# 			output = model(data)
# 			test_loss += criterion(output, target).item()
# 			pred = output.argmax(dim=1, keepdim=True)
# 			correct += pred.eq(target.view_as(pred)).sum().item()
# 	test_loss /= len(test_loader.dataset)
# 	print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function
def main():
	parser = argparse.ArgumentParser(description='Train PressureNet Model')
	parser.add_argument('--add_noise', type=float, default=0.1, help='amount of noise to add, 0 means no noise')
	parser.add_argument('--include_weight_height', action='store_true', default=False, help='include height and weight as input channels')
	parser.add_argument('--omit_contact_sobel', action='store_true', default=False, help='omit contact sobel')
	parser.add_argument('--use_hover', action='store_true', default=False, help='set depth_map_estimated_positive (channel 1) to 0')
	parser.add_argument('--mod', type=int, choices=[1, 2], required=True, help='choose a network (1 or 2)')
	parser.add_argument('--pmr', action='store_true', default=False, help='run PMR on input & precomputed spatial maps')
	parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
	args = parser.parse_args()

	config = {
		'add_noise':			args.add_noise,
		'include_weight_height':args.include_weight_height,
		'omit_contact_sobel':	args.omit_contact_sobel,
		'use_hover':			args.use_hover,
		'mod':					args.mod,
		'pmr':					args.pmr,
		'verbose':				args.verbose,

		# Not in args
		'batch_size':			1,
		'num_epochs':			1,
		'normalize_per_image':	True,
		'tanh':					True,
	}

	is_cuda_available = torch.cuda.is_available()
	device = torch.device("cuda" if is_cuda_available else "cpu")

	print(f"Device (CUDA/CPU):  {device}")
	print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
	print(f"Device Count:       {torch.cuda.device_count()}")
	print(f"Current Device:     {torch.cuda.current_device()}")

	# Set the np.array print options
	np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)


	# Define the path to the original train and test data (.pickle files)
	if config['mod'] == 1:
		train_files_dir = 'synthetic_data/original/train'
		test_files_dir = 'synthetic_data/original/test'
	elif config['mod'] == 2:
		train_files_dir = 'synthetic_data/by_mod1/a/train'
		test_files_dir = 'synthetic_data/by_mod1/a/test'

	# Get the paths to the train and test data files
	train_file_paths = retrieve_data_file_paths(train_files_dir)
	test_file_paths = retrieve_data_file_paths(test_files_dir)


	# Normalization standard deviations for each channel
	if config['normalize_per_image']:
		config['normalize_std_dev'] = np.ones(10, dtype=np.float32)
	else:
		config['normalize_std_dev'] = np.array([
    		1 / 41.8068,	# contact
			1 / 16.6955,	# pos est depth
			1 / 45.0851,	# neg est depth
			1 / 43.5580,	# cm est
			1 / 11.7015,	# pressure map
			1 / 45.6164 if config['add_noise'] else 1 / 29.8036,	# pressure map sobel
			1,  # OUTPUT DO NOTHING
			1,  # OUTPUT DO NOTHING
			1 / 30.2166,  # weight
			1 / 14.6293   # height
		])

	vertices = "all" if config['mod'] == 2 else [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

	# Define the parent array with -1 to indicate a non-existent parent (repaced 4294967295)
	parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

	train_dataset = PressurePoseDataset(file_paths=train_file_paths, config=config, is_train=True)
	test_dataset = PressurePoseDataset(file_paths=test_file_paths, config=config, is_train=False)
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

	# Define the model, optimizer, and loss function
	# model = PressureNet().to(device)
	# optimizer = optim.SGD(model.parameters(), lr=args.lr)
	# criterion = nn.CrossEntropyLoss()

	# Train and test the model
	# for epoch in range(1, args.epochs + 1):
	# 	train(model, device, train_loader, optimizer, criterion, epoch)
	# 	test(model, device, test_loader, criterion)

	############################################################################################################
	# SMPL model
	import torch
	import smplx


	# Paths to the smpl model pkl files
	# v1.0.0
	smpl_female_model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
	smpl_male_model_path = 'smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
	# v1.1.0
	smpl_female_model_path = 'smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'
	smpl_male_model_path = 'smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'

	# Check the GPU availability
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	smpl_female_model = smplx.SMPL(model_path=smpl_female_model_path).to(device)

	print(f'SMPL Female Model:')
	print(f'v_template:		{smpl_female_model.v_template.shape}')
	print(f'shapedirs:		{smpl_female_model.shapedirs.shape}')
	print(f'J_regressor:		{smpl_female_model.J_regressor.shape}')
	print(f'posedirs:		{smpl_female_model.posedirs.shape}')
	print(f'weights:		{smpl_female_model.lbs_weights.shape}')
	############################################################################################################

	for train_batch_idx, train_batch in enumerate(train_loader):
		inputs_batch, labels_batch = train_batch
		inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
		print(f"Train Batch Index:           {train_batch_idx + 1}")
		print(f"Train Batch Input Shape:     {inputs_batch.shape}")
		print(f"Train Batch Output Shape:    {labels_batch.shape}")
		print()

		# Plot the final input channels
		num_channels = inputs_batch.shape[1]
		num_cols = min(num_channels, 5)
		num_rows = (num_channels + num_cols - 1) // num_cols
		fig, axes = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))
		axes = axes.flatten() if num_rows > 1 else axes

		for i in range(num_channels):
			axes[i].imshow(inputs_batch[0, i].cpu().numpy())
			axes[i].set_title(f'Input Channel {i}\n{inputs_batch[0, i].shape} | {inputs_batch[0, i].dtype}\nmin: {inputs_batch[0, i].min():.2f} | max: {inputs_batch[0, i].max():.2f}\nmean: {inputs_batch[0, i].mean():.2f} | sum: {inputs_batch[0, i].sum():.2f}', fontsize=8, pad=10)
			axes[i].axis('off')

		for j in range(i + 1, len(axes)):
			axes[j].axis('off')

		plt.tight_layout()
		plt.show()
		print()



if __name__ == '__main__':
	main()
