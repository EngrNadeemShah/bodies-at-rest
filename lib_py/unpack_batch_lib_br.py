#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import pickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
# from scipy.misc import imresize
# from scipy.ndimage.interpolation import zoom

import sys
sys.path.insert(0, '../lib_py')

from kinematics_lib_br import KinematicsLib
from preprocessing_lib_br import PreprocessingLib

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



# import hrl_lib.util as ut
import pickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')

class UnpackBatchLib():


    def unpack_batch(self, batch, is_training, model, config):

        INPUT_DICT = {}
        adj_ext_idx = 0
        # 0:72: positions.
        batch.append(batch[1][:, 72:82])	# betas			| 'body_shape'
        batch.append(batch[1][:, 82:154])	# angles		| 'joint_angles'
        batch.append(batch[1][:, 154:157])	# root pos		| 'root_xyz_shift'
        batch.append(batch[1][:, 157:159])	# gender switch	| g1, g2
        batch.append(batch[1][:, 159])		# synth vs real	| s1
        batch.append(batch[1][:, 160:161])  # mass, kg		| 'body_mass'
        batch.append(batch[1][:, 161:162])  # height, kg	| 'body_height'

        if config['adjust_ang_from_est'] == True:
            adj_ext_idx += 3
            batch.append(batch[1][:, 162:172])	# 'betas_est'
            batch.append(batch[1][:, 172:244])	# 'angles_est'
            batch.append(batch[1][:, 244:247])	# 'root_xyz_est'
            if config['full_body_rot'] == True:
                adj_ext_idx += 1
                batch.append(batch[1][:, 247:253])	# 'root_atan2_est'

            extra_smpl_angles = batch[10]
            extra_targets = batch[11]
        else:
            extra_smpl_angles = None
            extra_targets = None


        if config['depth_map_labels'] and (config['depth_map_labels_test'] or is_training):
            batch.append(batch[0][:, config['num_input_channels_batch0'], : , :])       # mesh depth matrix
            batch.append(batch[0][:, config['num_input_channels_batch0'] + 1, : ,:])    # mesh contact matrix

            # cut off batch 0 so we don't have depth or contact on the input
            batch[0] = batch[0][:, :config['num_input_channels_batch0'], :, :]

        # cut it off so batch[2] is only the xyz marker targets
        batch[1] = batch[1][:, :72]     # markers_xyz_m

        x_images_ = batch[0].numpy()

        INPUT_DICT['x_images'] = np.copy(x_images_)


        #here perform synthetic calibration noise over pmat and sobel filtered pmat.
        if config['cal_noise'] == True:
            x_images_ = PreprocessingLib().preprocessing_add_calibration_noise(x_images_,
                                                                                          pmat_chan_idx = (config['num_input_channels_batch0']-2),
                                                                                          norm_std_coeffs = config['norm_std_coeffs'],
                                                                                          is_training = is_training,
                                                                                          noise_amount = config['cal_noise_amt'],
                                                                                          normalize_per_image = config['normalize_per_image'])


        if is_training == True: #only add noise to training images
            if config['cal_noise'] == False:
                x_images_ = PreprocessingLib().preprocessing_add_image_noise(np.array(x_images_),
                                                                                    pmat_chan_idx = (config['num_input_channels_batch0']-2),
                                                                                    norm_std_coeffs = config['norm_std_coeffs'])
            else:
                x_images_ = PreprocessingLib().preprocessing_add_image_noise(np.array(x_images_),
                                                                                    pmat_chan_idx = (config['num_input_channels_batch0']-1),
                                                                                    norm_std_coeffs = config['norm_std_coeffs'])

        x_images_ = PreprocessingLib().preprocessing_pressure_map_upsample(x_images_, multiple=2)

        x_images_ = np.array(x_images_)   # this line is added by Nadeem
        x_images = Variable(torch.Tensor(x_images_).type(config['dtype']), requires_grad=False)


        if config['incl_ht_wt_channels'] == True: #make images full of stuff
            weight_input = torch.ones((x_images.size()[0], x_images.size()[2] * x_images.size()[3])).type(config['dtype'])
            weight_input *= batch[7].type(config['dtype'])
            weight_input = weight_input.view((x_images.size()[0], 1, x_images.size()[2], x_images.size()[3]))
            height_input = torch.ones((x_images.size()[0], x_images.size()[2] * x_images.size()[3])).type(config['dtype'])
            height_input *= batch[8].type(config['dtype'])
            height_input = height_input.view((x_images.size()[0], 1, x_images.size()[2], x_images.size()[3]))
            x_images = torch.cat((x_images, weight_input, height_input), 1)


        y_true_markers_xyz, y_true_betas = Variable(batch[1].type(config['dtype']), requires_grad=False), \
                         Variable(batch[2].type(config['dtype']), requires_grad=False)

        y_true_angles = Variable(batch[3].type(config['dtype']), requires_grad=is_training)
        y_true_root_xyz = Variable(batch[4].type(config['dtype']), requires_grad=is_training)
        y_true_gender_switch = Variable(batch[5].type(config['dtype']), requires_grad=is_training)
        y_true_synth_real_switch = Variable(batch[6].type(config['dtype']), requires_grad=is_training)

        OUTPUT_EST_DICT = {}
        if config['adjust_ang_from_est'] == True:     # False in our case, therefore OUTPUT_EST_DICT remains empty
            OUTPUT_EST_DICT['betas'] = Variable(batch[9].type(config['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['angles'] = Variable(extra_smpl_angles.type(config['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['root_shift'] = Variable(extra_targets.type(config['dtype']), requires_grad=is_training)
            if config['full_body_rot'] == True:
                OUTPUT_EST_DICT['root_atan2'] = Variable(batch[12].type(config['dtype']), requires_grad=is_training)

        if config['depth_map_labels'] == True:        # False in our case
            if config['depth_map_labels_test'] == True or is_training == True:
                INPUT_DICT['batch_mdm'] = batch[9+adj_ext_idx].type(config['dtype'])
                INPUT_DICT['batch_cm'] = batch[10+adj_ext_idx].type(config['dtype'])
        else:
            INPUT_DICT['batch_mdm'] = None
            INPUT_DICT['batch_cm'] = None


        if config['omit_cntct_sobel'] == True:        # False in our case
            x_images[:, 0, :, :] *= 0

            if config['cal_noise'] == True:
                x_images[:, config['num_input_channels_batch0'], :, :] *= 0
            else:
                x_images[:, config['num_input_channels_batch0']-1, :, :] *= 0


        if config['use_hover'] == False and config['adjust_ang_from_est'] == True:      # False in our case
            x_images[:, 1, :, :] *= 0


        scores, OUTPUT_DICT = model.forward_kinematic_angles(x_images = x_images,
                                                             y_true_markers_xyz = y_true_markers_xyz,
                                                             y_true_betas = y_true_betas,
                                                             y_true_angles = y_true_angles,
                                                             y_true_root_xyz = y_true_root_xyz,
                                                             y_true_gender_switch = y_true_gender_switch,
                                                             y_true_synth_real_switch = y_true_synth_real_switch,
                                                             CTRL_PNL = config,
                                                             OUTPUT_EST_DICT = OUTPUT_EST_DICT,
                                                             is_training = is_training,
                                                             )  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.


        INPUT_DICT['x_images'] = x_images.data
        INPUT_DICT['y_true_markers_xyz'] = y_true_markers_xyz.data

        return scores, INPUT_DICT, OUTPUT_DICT

