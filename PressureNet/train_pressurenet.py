import sys
import os
import time
import numpy as np
import chumpy as ch
import optparse
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import convnet_br as convnet
from datasets import DatasetClass


# Add the lib_py directory to the system path (temporarily) to import the custom libraries
sys.path.insert(0, '../lib_py')

# Pose Estimation Libraries
from visualization_lib_br import VisualizationLib
from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib
from unpack_batch_lib_br import UnpackBatchLib


# Constants
MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286  # metres


# Set the np.array print options
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

# Set the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the number of threads to be used by PyTorch (useful for debugging)
# torch.set_num_threads(1)

print(f"Device (CUDA/CPU):  {device}")
if torch.cuda.is_available():
	print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
	print(f"Device Count:       {torch.cuda.device_count()}")
	print(f"Current Device:     {torch.cuda.current_device()}")


def load_pickle(filename):
	with open(filename, 'rb') as f:
		return pkl.load(f, encoding='latin1')


class PhysicalTrainer():
	'''Gets the dictionary of pressure maps from the training database,
	and will have API to do all sorts of training with it.'''

	def __init__(self, train_filepaths_f, train_filepaths_m, test_filepaths_f, test_filepaths_m, cmd_args):
		'''Opens the specified pickle files to get the combined dataset:
		This dataset is a dictionary of pressure maps with the corresponding
		3d position and orientation of the markers associated with it.'''

		# Set the configuration parameters (some from the command line args)
		self.cmd_args = cmd_args
		self.config = {
			# Training Parameters (6)
			'batch_size':            16,		# 128 changed by Nadeem to 512
			'num_epochs':            1,		# 100 changed by Nadeem to 1
			'shuffle':               True,
			'loss_root':             cmd_args.loss_root,
			'double_network_size':   False,
			'first_pass':            True,

			# Input Channels (5)
			'incl_ht_wt_channels':   cmd_args.htwt,
			'omit_cntct_sobel':      cmd_args.omit_cntct_sobel,
			'use_hover':             cmd_args.use_hover,
			'num_input_channels':    2,

			# Preprocessing (6)
			'normalize_per_image':   True,
			'clip_sobel':            True,
			'clip_betas':            True,
			'cal_noise':             cmd_args.cal_noise,
			'cal_noise_amt':         0.1,	# 0.2 changed by Nadeem to 0.1
			'mesh_bottom_dist':      True,

			# Output (Labels) (7)
			'depth_map_labels':      cmd_args.pmr,		# can only be true if we have 100% synthetic data for training
			'depth_map_labels_test': cmd_args.pmr,		# can only be true if we have 100% synth for testing
			'depth_map_output':      cmd_args.pmr,
			'depth_map_input_est':   cmd_args.pmr,		# do this if we're working in a two-part regression
			'regress_angles':        cmd_args.regress_angles,
			'full_body_rot':         True,
			'align_procr':           False,

			# Miscellaneous (7)
			'loss_type':             cmd_args.loss_type,
			'verbose':               cmd_args.verbose,
			'lock_root':             False,
			'GPU':                   torch.cuda.is_available(),
			'dtype':                 torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
			'all_tanh_activ':        True,
			'pmat_mult':             1,
		}


		if cmd_args.mod == 1:
			self.config['adjust_ang_from_est'] = False  # Starts angles from scratch
		elif cmd_args.mod == 2:
			self.config['adjust_ang_from_est'] = True   # Gets betas and angles from prior estimate

		if self.config['cal_noise']:	# If we're adding calibration noise, we need to recompute the pressure map contact mask every time
			# Old flag: config['incl_pmat_cntct_input'] is replaced by not config['cal_noise']
			self.config['clip_sobel'] = False

		self.config['num_input_channels'] += int(not self.config['cal_noise'])
		self.config['num_input_channels'] += 3 if self.config['depth_map_input_est'] else 0
		self.config['num_input_channels_batch0'] = self.config['num_input_channels']
		self.config['num_input_channels'] += 2 if self.config['incl_ht_wt_channels'] else 0
		self.config['num_input_channels'] += int(self.config['cal_noise'])


		# Standard deviations for different multipliers
		pmat_std_from_mult = np.array([None, 11.7015, 19.9091, 23.0702, 0.0, 25.5054])
		sobel_std_from_mult = np.array([None, 29.8036, 33.3353, 34.1443, 0.0, 34.8639]) if not self.config['cal_noise'] else np.array([None, 45.6164, 77.7492, 88.8940, 0.0, 97.9008])


		# Normalization standard coefficients
		if self.config['normalize_per_image']:
			self.config['norm_std_coeffs'] = np.ones(10, dtype=np.float32)
		else:
			self.config['norm_std_coeffs'] = np.array([
				1 / 41.8068,  # contact
				1 / 16.6955,  # pos est depth
				1 / 45.0851,  # neg est depth
				1 / 43.5580,  # cm est
				1 / pmat_std_from_mult[self.config['pmat_mult']],  # pmat x5
				1 / sobel_std_from_mult[self.config['pmat_mult']],  # pmat sobel
				1,  # OUTPUT DO NOTHING
				1,  # OUTPUT DO NOTHING
				1 / 30.2166,  # weight
				1 / 14.6293   # height
			], dtype=np.float32)

		self.config['convnet_fp_prefix'] = '../data_BR/convnets/'

		self.vertices = "all" if self.config['depth_map_output'] else [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

		self.weight_depth_planes = (1 - cmd_args.j_d_ratio)     # *2

		self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
		self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
		self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)


		# Load the training dataset
		train_filepaths = train_filepaths_f + train_filepaths_m

		self.train_dataset = DatasetClass(
			file_list       = train_filepaths,
			creation_type   = 'synth',
			test            = False,
			config			= self.config,
			mat_size        = self.mat_size,
			blur_sigma      = 0.5,
			verbose		 	= self.config['verbose'],
			)

		print(f"Number of Total Train Samples:  {len(self.train_dataset)}")
		print(f"Batch Size:                     {self.config['batch_size']}")
		print(f"Number of Iterations:           {len(self.train_dataset) // self.config['batch_size'] + 1}")

		# Create a DataLoader object for the training dataset
		self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=False)

		# Load the testing dataset
		test_filepaths = test_filepaths_f + test_filepaths_m

		self.test_dataset = DatasetClass(
			file_list       = test_filepaths,
			creation_type   = 'synth',
			test            = True,
			config			= self.config,
			mat_size        = self.mat_size,
			blur_sigma      = 0.5,
			verbose			= self.config['verbose'],
			)

		print(f"Number of Total Test Samples:   {len(self.test_dataset)}")
		print(f"Batch Size:                     {self.config['batch_size']}")
		print(f"Number of Iterations:           {len(self.test_dataset) // self.config['batch_size'] + 1}")

		# Create a DataLoader object for the testing dataset
		self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False)


		# Generate the save name for the model
		self.save_name = f"_{cmd_args.mod}_{cmd_args.loss_type}_{len(self.train_dataset)}ct_{self.config['batch_size']}b_x{self.config['pmat_mult']}pm"

		if self.config['depth_map_labels']:		self.save_name += f'_{self.cmd_args.j_d_ratio}rtojtdpth'
		if self.config['depth_map_input_est']:	self.save_name += '_depthestin'
		if self.config['adjust_ang_from_est']:	self.save_name += '_angleadj'
		if self.config['all_tanh_activ']:		self.save_name += '_tnh'
		if self.config['incl_ht_wt_channels']:	self.save_name += '_htwt'
		if self.config['cal_noise']:			self.save_name += f'_clns{int(self.config["cal_noise_amt"] * 100)}p'
		if self.config['double_network_size']:	self.save_name += '_dns'
		if self.config['loss_root']:			self.save_name += '_rt'
		if self.config['omit_cntct_sobel']:		self.save_name += '_ocs'
		if self.config['use_hover']:			self.save_name += '_uh'
		if self.cmd_args.half_shape_wt:			self.save_name += '_hsw'

		print(f'appending to train{self.save_name}')

		self.train_val_losses = {
			'train_loss': [],
			'val_loss': [],
			'epoch_ct': []
		}


		# Add .to(device) to all tensors
		# for test_batch_idx, test_batch in enumerate(self.test_loader):
		#     print(f"Test Batch Index:           {test_batch_idx}")
		#     print(f"Test Batch Input Shape:     {test_batch[0].shape}")
		#     print(f"Test Batch Output Shape:    {test_batch[1].shape}")
		#     print()
		#     break


	def init_convnet_train(self):
		convnet_fc_output_size = 85  # 10 + 3 + 24*3 --- betas, root shift, rotations

		if self.config['full_body_rot']:
			convnet_fc_output_size += 3

		if self.cmd_args.go200:
			model_path = f"{self.config['convnet_fp_prefix']}convnet_1_anglesDC_184000ct_128b_x1pm_tnh_clns20p_100e_2e-05lr.pt"
			self.model = torch.load(model_path)

		else:
			in_channels = self.config['num_input_channels'] - 2 if self.cmd_args.omit_cntct_sobel else self.config['num_input_channels']
			self.model = convnet.CNN(
				convnet_fc_output_size,
				self.config['loss_type'],
				vertices = self.vertices,
				in_channels = in_channels
				)

		total_params = sum(p.numel() for p in self.model.parameters())
		print(f"\nTotal number of parameters: {total_params}")

		# Move the model to the GPU if available
		self.model = self.model.to(device)

		learning_rate = 0.00002

		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0005) # start with .00005

		print(f"Train and Validation Losses: {self.train_val_losses}")

		# Train the model for the specified number of epochs
		for epoch in range(1, self.config['num_epochs'] + 1):
			start_time = time.perf_counter()
			self.train_convnet(epoch)

			elapsed_time = time.perf_counter() - start_time
			print(f'Time taken by epoch {epoch}: {elapsed_time:.2f} seconds')

			# Save the model and losses every 10 epochs
			if epoch % 10 == 0 or epoch == self.config['num_epochs']:
				epoch_log = epoch + 100 if self.cmd_args.go200 else epoch

				print("Saving convnet.")
				model_path = f"{self.config['convnet_fp_prefix']}convnet{self.save_name}_{epoch_log}e_{learning_rate}lr.pt"
				torch.save(self.model, model_path)
				print("Saved convnet.")

				losses_path = f"{self.config['convnet_fp_prefix']}convnet_losses{self.save_name}_{epoch_log}e_{learning_rate}lr.p"
				with open(losses_path, 'wb') as f:
					pkl.dump(self.train_val_losses, f)
				print("Saved losses.")


	def train_convnet(self, epoch):
		'''
		Train the model for one epoch.
		'''

		self.model.train()
		self.criterion = nn.L1Loss()
		self.criterion2 = nn.MSELoss()
		with torch.autograd.set_detect_anomaly(True):

			# This will loop a total = training_images/batch_size times
			for train_batch_idx, train_batch in enumerate(self.train_loader):
				print(f"Train Batch Index:           {train_batch_idx}")
				print(f"Train Batch Input Shape:     {train_batch[0].shape}")
				print(f"Train Batch Output Shape:    {train_batch[1].shape}")
				print()

				inputs, targets = train_batch
				inputs, targets = inputs.to(device), targets.to(device)

				self.optimizer.zero_grad()
				scores, INPUT_DICT, OUTPUT_DICT = \
					UnpackBatchLib().unpack_batch(train_batch, is_training=True, model = self.model, config=self.config)

				self.config['first_pass'] = False

				scores_zeros = torch.zeros((inputs.shape[0], scores.size()[1]), device=device, requires_grad=True)

				if self.config['full_body_rot'] == True:
					OSA = 6
					if self.cmd_args.loss_root == True:
						loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16])
					else:
						loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0
				else: OSA = 0

				loss_eucl = self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:, 10+OSA:34+OSA])
				if self.cmd_args.half_shape_wt == True:
					loss_betas = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * 0.5
				else:
					loss_betas = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10])


				if self.config['regress_angles'] == True:
					loss_angs = self.criterion2(scores[:, 34+OSA:106+OSA], scores_zeros[:, 34+OSA:106+OSA])
					loss = (loss_betas + loss_eucl + loss_bodyrot + loss_angs)
				else:
					loss = (loss_betas + loss_eucl + loss_bodyrot)


				#print INPUT_DICT['depth_map'].size(), OUTPUT_DICT['batch_mdm_est'].size()
				if self.config['depth_map_labels'] == True:
					hover_map = OUTPUT_DICT['batch_mdm_est'].clone()
					hover_map[hover_map < 0] = 0

					INPUT_DICT['depth_map'][INPUT_DICT['depth_map'] > 0] = 0
					if self.config['mesh_bottom_dist'] == True:
						OUTPUT_DICT['batch_mdm_est'][OUTPUT_DICT['batch_mdm_est'] > 0] = 0

					loss_mesh_depth = self.criterion2(INPUT_DICT['depth_map'], OUTPUT_DICT['batch_mdm_est'])*self.weight_depth_planes * (1. / 44.46155340000357) * (1. / 44.46155340000357)
					loss_mesh_contact = self.criterion(INPUT_DICT['contact_map'], OUTPUT_DICT['batch_cm_est'])*self.weight_depth_planes * (1. / 0.4428100696329912)

					loss += loss_mesh_depth
					loss += loss_mesh_contact



				loss.backward()
				self.optimizer.step()
				loss *= 1000


				if train_batch_idx % cmd_args.log_interval == 0:# and batch_idx > 0:


					val_n_batches = 4
					print(f"Evaluating on {val_n_batches} batches")

					im_display_idx = 0 #random.randint(0,INPUT_DICT['x_images'].size()[0])


					if self.config['GPU']:
						VisualizationLib().print_error_train(INPUT_DICT['y_true_markers_xyz'].cpu(), OUTPUT_DICT['y_pred_markers_xyz'].cpu(),
															 self.output_size_train, data='train')
					else:
						VisualizationLib().print_error_train(INPUT_DICT['y_true_markers_xyz'], OUTPUT_DICT['y_pred_markers_xyz'],
															 self.output_size_train, data='train')

				   # print INPUT_DICT['x_images'][im_display_idx, 4:, :].type()

					if self.config['depth_map_labels'] == True: #pmr regression
						self.cntct_in = INPUT_DICT['x_images'][im_display_idx, 0, :].squeeze().cpu()/self.config['norm_std_coeffs'][0]  #contact
						self.pimage_in = INPUT_DICT['x_images'][im_display_idx, 1, :].squeeze().cpu()/self.config['norm_std_coeffs'][4] #pmat
						self.sobel_in = INPUT_DICT['x_images'][im_display_idx, 2, :].squeeze().cpu()/self.config['norm_std_coeffs'][5]  #sobel
						self.pmap_recon = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze()*-1).cpu().data #est depth output
						self.cntct_recon = (OUTPUT_DICT['batch_cm_est'][im_display_idx, :, :].squeeze()).cpu().data #est depth output
						self.hover_recon = (hover_map[im_display_idx, :, :].squeeze()).cpu().data #est depth output
						self.pmap_recon_gt = (INPUT_DICT['depth_map'][im_display_idx, :, :].squeeze()*-1).cpu().data #ground truth depth
						self.cntct_recon_gt = (INPUT_DICT['contact_map'][im_display_idx, :, :].squeeze()).cpu().data #ground truth depth
					else:
						self.cntct_in = INPUT_DICT['x_images'][im_display_idx, 0, :].squeeze().cpu()/self.config['norm_std_coeffs'][0]  #contact
						self.pimage_in = INPUT_DICT['x_images'][im_display_idx, 1, :].squeeze().cpu()/self.config['norm_std_coeffs'][4]  #pmat
						self.sobel_in = INPUT_DICT['x_images'][im_display_idx, 2, :].squeeze().cpu()/self.config['norm_std_coeffs'][5]  #sobel
						self.pmap_recon = None
						self.cntct_recon = None
						self.hover_recon = None
						self.pmap_recon_gt = None
						self.cntct_recon_gt = None

					if self.config['depth_map_input_est'] == True: #this is a network 2 option ONLY
						self.pmap_recon_in = INPUT_DICT['x_images'][im_display_idx, 2, :].squeeze().cpu()/self.config['norm_std_coeffs'][2] #pmat
						self.cntct_recon_in = INPUT_DICT['x_images'][im_display_idx, 3, :].squeeze().cpu()/self.config['norm_std_coeffs'][3] #pmat
						self.hover_recon_in = INPUT_DICT['x_images'][im_display_idx, 1, :].squeeze().cpu()/self.config['norm_std_coeffs'][1] #pmat
						self.pimage_in = INPUT_DICT['x_images'][im_display_idx, 4, :].squeeze().cpu()/self.config['norm_std_coeffs'][4] #pmat
						self.sobel_in = INPUT_DICT['x_images'][im_display_idx, 5, :].squeeze().cpu()/self.config['norm_std_coeffs'][5]  #sobel
					else:
						self.pmap_recon_in = None
						self.cntct_recon_in = None
						self.hover_recon_in = None




					self.tar_sample = INPUT_DICT['y_true_markers_xyz']
					self.tar_sample = self.tar_sample[im_display_idx, :].squeeze() / 1000
					self.sc_sample = OUTPUT_DICT['y_pred_markers_xyz'].clone()
					self.sc_sample = self.sc_sample[im_display_idx, :].squeeze() / 1000
					self.sc_sample = self.sc_sample.view(self.output_size_train)

					train_loss = loss.data.item()
					examples_this_epoch = train_batch_idx * len(INPUT_DICT['x_images'])
					epoch_progress = 100. * train_batch_idx / len(self.train_loader)

					val_loss = self.validate_convnet(n_batches=val_n_batches)


					print_text_list = [ 'Train Epoch: {} ',
										'[{}',
										'/{} ',
										'({:.0f}%)]\t']
					print_vals_list = [epoch,
									  examples_this_epoch,
									  len(self.train_loader.dataset),
									  epoch_progress]
					if self.config['loss_type'] == 'anglesR' or self.config['loss_type'] == 'anglesDC' or self.config['loss_type'] == 'anglesEU':
						print_text_list.append('Train Loss Joints: {:.2f}')
						print_vals_list.append(1000*loss_eucl.data)
						print_text_list.append('\n\t\t\t\t\t\t   Betas Loss: {:.2f}')
						print_vals_list.append(1000*loss_betas.data)
						if self.config['full_body_rot'] == True:
							print_text_list.append('\n\t\t\t\t\t\tBody Rot Loss: {:.2f}')
							print_vals_list.append(1000*loss_bodyrot.data)
						if self.config['regress_angles'] == True:
							print_text_list.append('\n\t\t\t\t\t\t  Angles Loss: {:.2f}')
							print_vals_list.append(1000*loss_angs.data)
						if self.config['depth_map_labels'] == True:
							print_text_list.append('\n\t\t\t\t\t\t   Mesh Depth: {:.2f}')
							print_vals_list.append(1000*loss_mesh_depth.data)
							print_text_list.append('\n\t\t\t\t\t\t Mesh Contact: {:.2f}')
							print_vals_list.append(1000*loss_mesh_contact.data)

					print_text_list.append('\n\t\t\t\t\t\t   Total Loss: {:.2f}')
					print_vals_list.append(train_loss)

					print_text_list.append('\n\t\t\t\t\t  Val Total Loss: {:.2f}')
					print_vals_list.append(val_loss)



					print_text = ''
					for item in print_text_list:
						print_text += item
					print((print_text.format(*print_vals_list)))


					print('appending to alldata losses')
					self.train_val_losses['train_loss'].append(train_loss)
					self.train_val_losses['epoch_ct'].append(epoch)
					self.train_val_losses['val_loss'].append(val_loss)


	def validate_convnet(self, n_batches=None):
		self.model.eval()
		loss = 0.
		n_examples = 0
		batch_ct = 1

		with torch.no_grad():
			for batch_i, batch in enumerate(self.test_loader):
				inputs, targets = batch
				inputs, targets = inputs.to(device), targets.to(device)

				scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
					UnpackBatchLib().unpack_batch(batch, is_training=False, model=self.model, config=self.config)
				scores_zeros = torch.zeros((inputs.shape[0], scores.size()[1]), device=device)

				loss_to_add = 0

				if self.config['full_body_rot'] == True:
					OSA = 6
					if self.cmd_args.loss_root == True:
						loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]))
					else:
						loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0)
					loss_to_add += loss_bodyrot
				else: OSA = 0

				loss_eucl = float(self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:,  10+OSA:34+OSA]))
				if self.cmd_args.half_shape_wt == True:
					loss_betas = float(self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * 0.5)
				else:
					loss_betas = float(self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]))



				if self.config['regress_angles'] == True:
					loss_angs = float(self.criterion(scores[:, 34+OSA:106+OSA], scores_zeros[:, 34+OSA:106+OSA]))
					loss_to_add += (loss_betas + loss_eucl + loss_angs)
				else:
					loss_to_add += (loss_betas + loss_eucl)

				# print INPUT_DICT_VAL['batch_mdm'].size(), OUTPUT_DICT_VAL['batch_mdm_est'].size()

				if self.config['depth_map_labels'] == True:
					INPUT_DICT_VAL['batch_mdm'][INPUT_DICT_VAL['batch_mdm'] > 0] = 0
					if self.config['mesh_bottom_dist'] == True:
						OUTPUT_DICT_VAL['batch_mdm_est'][OUTPUT_DICT_VAL['batch_mdm_est'] > 0] = 0

					loss_mesh_depth = float(self.criterion2(INPUT_DICT_VAL['batch_mdm'],OUTPUT_DICT_VAL['batch_mdm_est']) * self.weight_depth_planes * (1. / 44.46155340000357) * (1. / 44.46155340000357))
					loss_mesh_contact = float(self.criterion(INPUT_DICT_VAL['batch_cm'],OUTPUT_DICT_VAL['batch_cm_est']) * self.weight_depth_planes * (1. / 0.4428100696329912))

					loss_to_add += loss_mesh_depth
					loss_to_add += loss_mesh_contact

				loss += loss_to_add

				#print loss
				n_examples += self.config['batch_size']

				if n_batches and (batch_i >= n_batches):
					break

				batch_ct += 1
				#break


			loss /= batch_ct
			loss *= 1000

		if self.cmd_args.visualize == True:
			VisualizationLib().visualize_pressure_map(pimage_in = self.pimage_in, cntct_in = self.cntct_in, sobel_in = self.sobel_in,
													  targets_raw = self.tar_sample.cpu(), scores_net1 = self.sc_sample.cpu(),
													  pmap_recon_in = self.pmap_recon_in, cntct_recon_in = self.cntct_recon_in,
													  hover_recon_in = self.hover_recon_in,
													  pmap_recon = self.pmap_recon, cntct_recon = self.cntct_recon, hover_recon = self.hover_recon,
													  pmap_recon_gt=self.pmap_recon_gt, cntct_recon_gt = self.cntct_recon_gt,
													  block=False)

		#print "loss is:" , loss
		return loss






if __name__ == "__main__":

	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Train PressureNet Model.')

	parser.add_argument('--cal_noise', action='store_true', default=False,
						help='Apply calibration noise to the input to facilitate sim to real transfer.')

	parser.add_argument('--go200', action='store_true', default=False,
						help='Run network 1 for 100 to 200 epochs.')

	parser.add_argument('--half_shape_wt', action='store_true', default=False,
						help='Half betas.')

	parser.add_argument('--hard_disk', action='store_true', default=False,
						help='Read and write to data on an external harddrive.')

	parser.add_argument('--htwt', action='store_true', default=False,
						help='Include height and weight info on the input.')

	parser.add_argument('--j_d_ratio', type=float, default=0.5,
						help='Set the loss mix: joints to depth planes. Only used for PMR regression. PMR parameter to adjust loss function 2.')

	parser.add_argument('--log_interval', type=int, default=100, metavar='N',
						help='Number of batches between logging train status. Note: Visualizing too often can slow down training.')

	parser.add_argument('--loss_root', action='store_true', default=False,
						help='Use root in loss function.')

	parser.add_argument('--loss_type', type=str, default='anglesDC',
						help='Choose direction cosine or euler angle regression.')

	parser.add_argument('--mod', type=int, default=0,
						help='Choose a network.')

	parser.add_argument('--omit_cntct_sobel', action='store_true', default=False,
						help='Cut contact and sobel from input.')

	parser.add_argument('--pmr', action='store_true', default=False,
						help='Run PMR on input plus precomputed spatial maps.')

	parser.add_argument('--qt', action='store_true', dest='quick_test', default=False,
						help='Do a quick test.')

	parser.add_argument('--regress_angles', action='store_true', default=False,  # I found this option doesn't help much.
						help='Regress the angles as well as betas and joint pos.')

	parser.add_argument('--use_hover', action='store_true', default=False,
						help='Cut hovermap from pmr input.')

	parser.add_argument('--verbose', action='store_true', default=False,
						help='Printout everything (under construction).')

	parser.add_argument('--visualize', action='store_true', default=False,
						help='Visualize training.')

	cmd_args = parser.parse_args()


	# Create path to data files
	filepaths_prefix = "/media/henry/multimodal_data_2/data_BR/" if cmd_args.hard_disk else "../data_BR/"
	filepaths_suffix = ''

	if cmd_args.mod == 2 or cmd_args.quick_test:
		filepaths_suffix = f"_convnet_1_{cmd_args.loss_type}_184000ct_128b_x1pm_tnh"
		if cmd_args.htwt:
			filepaths_suffix += '_htwt'
		if cmd_args.cal_noise:
			filepaths_suffix += '_clns10p'    # changed from _clns20p to _clns10p by Nadeem
		if cmd_args.loss_root:
			filepaths_suffix += '_rt'
		if cmd_args.omit_cntct_sobel:
			filepaths_suffix += '_ocs'
		if cmd_args.half_shape_wt:
			filepaths_suffix += '_hsw'
		filepaths_suffix += f'_100e_{0.00002}lr'

	elif cmd_args.mod == 1:
		filepaths_suffix = ''

	else:
		print("Please choose a valid network. You can specify '--net 1' or '--net 2'.")
		sys.exit()


	train_filepaths_f = []
	train_filepaths_m = []
	test_filepaths_f = []
	test_filepaths_m = []


	if cmd_args.quick_test:
		# Run a quick test
		train_filepaths_f.append(f"{filepaths_prefix}synth/quick_test/test_rollpi_f_lay_set23to24_3000_qt{filepaths_suffix}.p")
		test_filepaths_f.append(f"{filepaths_prefix}synth/quick_test/test_rollpi_f_lay_set23to24_3000_qt{filepaths_suffix}.p")

	else:
		# General partition - 104,000 train + 12,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/general/train_rollpi_f_lay_set18to22_10000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general/train_rollpi_plo_f_lay_set18to22_10000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general/train_rollpi_m_lay_set18to22_10000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general/train_rollpi_plo_m_lay_set18to22_10000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general/train_rollpi_f_lay_set10to17_16000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general/train_rollpi_plo_f_lay_set10to17_16000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general/train_rollpi_m_lay_set10to17_16000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general/train_rollpi_plo_m_lay_set10to17_16000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/general/test_rollpi_f_lay_set23to24_3000{filepaths_suffix}.p")
		test_filepaths_f.append(f"{filepaths_prefix}synth/general/test_rollpi_plo_f_lay_set23to24_3000{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/general/test_rollpi_m_lay_set23to24_3000{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/general/test_rollpi_plo_m_lay_set23to24_3000{filepaths_suffix}.p")


		# General supine partition - 52,000 train + 6,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/train_roll0_f_lay_set5to7_5000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/train_roll0_plo_f_lay_set5to7_5000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/train_roll0_m_lay_set5to7_5000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/train_roll0_plo_m_lay_set5to7_5000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/train_roll0_f_lay_set10to13_8000{filepaths_suffix}.p")
		train_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/train_roll0_plo_f_lay_set10to13_8000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/train_roll0_m_lay_set10to13_8000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/train_roll0_plo_m_lay_set10to13_8000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/test_roll0_f_lay_set14_1500{filepaths_suffix}.p")
		test_filepaths_f.append(f"{filepaths_prefix}synth/general_supine/test_roll0_plo_f_lay_set14_1500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/test_roll0_m_lay_set14_1500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/general_supine/test_roll0_plo_m_lay_set14_1500{filepaths_suffix}.p")


		# Hands behind head partition - 4,000 train + 1,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/hands_behind_head/train_roll0_plo_hbh_f_lay_set1to2_2000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/hands_behind_head/train_roll0_plo_hbh_m_lay_set2pa1_2000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/hands_behind_head/test_roll0_plo_hbh_f_lay_set4_500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/hands_behind_head/test_roll0_plo_hbh_m_lay_set1_500{filepaths_suffix}.p")


		# Prone hands up partition - 8,000 train + 1,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/prone_hands_up/train_roll0_plo_phu_f_lay_set2pl4_4000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/prone_hands_up/train_roll0_plo_phu_m_lay_set2pl4_4000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/prone_hands_up/test_roll0_plo_phu_f_lay_set1pa3_500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/prone_hands_up/test_roll0_plo_phu_m_lay_set1pa3_500{filepaths_suffix}.p")


		# Straight limbs partition - 8,000 train + 1,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/straight_limbs/test_roll0_sl_f_lay_set1both_500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/straight_limbs/test_roll0_sl_m_lay_set1both_500{filepaths_suffix}.p")


		# Crossed legs partition - 8,000 train + 1,000 test
		train_filepaths_f.append(f"{filepaths_prefix}synth/crossed_legs/train_roll0_xl_f_lay_set2both_4000{filepaths_suffix}.p")
		train_filepaths_m.append(f"{filepaths_prefix}synth/crossed_legs/train_roll0_xl_m_lay_set2both_4000{filepaths_suffix}.p")

		test_filepaths_f.append(f"{filepaths_prefix}synth/crossed_legs/test_roll0_xl_f_lay_set1both_500{filepaths_suffix}.p")
		test_filepaths_m.append(f"{filepaths_prefix}synth/crossed_legs/test_roll0_xl_m_lay_set1both_500{filepaths_suffix}.p")


	p = PhysicalTrainer(train_filepaths_f, train_filepaths_m, test_filepaths_f, test_filepaths_m, cmd_args)

	p.init_convnet_train()