import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel, zoom
from random import normalvariate
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import torch.nn.functional as F
mpl.rcParams['text.usetex'] = False  # Disable LaTeX rendering
mpl.rcParams['font.family'] = 'DejaVu Sans'  # Set default font family
mpl.use('Agg')
# mpl.use('TkAgg')

def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f, encoding='latin1')

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, split='train', transform=None, resize_factor=1.0, verbose=False):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            split (str): 'train' or 'test' to load the respective dataset.
            transform (callable, optional): Optional transform to apply to inputs.
            verbose (bool): If True, print additional information about the dataset.
        """
        self.hdf5_file_path = hdf5_file_path
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.resize_factor = resize_factor
        
        # Open the file to get keys
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            self.groups = []
            self.lengths = []
            
            for key in hdf5_file[split]:  # Iterate over different body postures
                for gender in hdf5_file[f'{split}/{key}']:  # Iterate over male/female
                    inputs_path = f'{split}/{key}/{gender}/inputs'
                    labels_path = f'{split}/{key}/{gender}/labels'
                    
                    if inputs_path in hdf5_file and labels_path in hdf5_file:
                        self.groups.append((inputs_path, labels_path))
                        self.lengths.append(hdf5_file[inputs_path].shape[0])

            self.cumulative_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0)
            self.total_size = self.cumulative_lengths[-1].item()
            self.num_channels = hdf5_file[inputs_path].shape[1]
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            # Find the right dataset based on cumulative lengths
            # dataset_idx = next(i for i, length in enumerate(self.cumulative_lengths) if idx < length)
            for i, length in enumerate(self.cumulative_lengths):
                if idx < length:
                    dataset_idx = i
                    break
            
            if dataset_idx > 0:
                idx = idx - self.cumulative_lengths[dataset_idx - 1].item()
            
            inputs_path, labels_path = self.groups[dataset_idx]
            
            input_data = hdf5_file[inputs_path][idx]
            label_data = hdf5_file[labels_path][idx]
            
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            if self.resize_factor < 1.0:
                input_tensor = F.interpolate(input_tensor.unsqueeze(0), 
                                            scale_factor=self.resize_factor, 
                                            mode='bilinear', align_corners=False).squeeze(0)

            label_tensor = torch.tensor(label_data, dtype=torch.float32)
            
            if self.transform:
                input_tensor = self.transform(input_tensor)
            
            if self.verbose:
                print(f"inputs_path: {inputs_path}, labels_path: {labels_path}, idx: {idx}, dataset_idx: {dataset_idx}")
            return input_tensor, label_tensor