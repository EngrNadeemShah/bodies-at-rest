from torch.utils.data import Dataset, DataLoader
import torch
import os

class PressurePoseProcessedDataset(Dataset):
	def __init__(self, data_dir):
		"""Load preprocessed .pt files."""
		self.data_dir = data_dir
		self.file_paths = self._create_and_sort_file_paths()
		self.data_indices = self._index_data()

	def _create_and_sort_file_paths(self):
		"""
		Create a list of file paths and sort them.
		"""
		file_paths = []
		for dir_path, _, file_names in os.walk(self.data_dir):
			for file_name in file_names:
				if not file_name.endswith('.pt'):
					continue
				file_path = os.path.join(dir_path, file_name)
				file_paths.append(file_path)
		return sorted(file_paths)

	def _index_data(self):
		"""
		Index the data across all files without loading it into memory.
		Returns a list of tuples (file_path, data_idx).
		"""
		data_indices = []
		for file_path in self.file_paths:
			data_dict = torch.load(file_path)

			# Index the data based on the key 'labels'
			for data_idx in range(len(data_dict['labels'])):
				data_indices.append((file_path, data_idx))

		print(f'data_indices: {data_indices}')
		return data_indices

	def __len__(self):
		return len(self.data_indices)

	def __getitem__(self, idx):
		file_path, data_idx = self.data_indices[idx]
		data_dict = torch.load(file_path)
		return data_dict['inputs'][data_idx], data_dict['labels'][data_idx]