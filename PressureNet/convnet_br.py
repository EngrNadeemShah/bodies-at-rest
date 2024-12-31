import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.stats as ss
import torchvision
import time

import sys
sys.path.insert(0, '../lib_py')
# import os

# # Add the parent directory of lib_py to sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib_py')))


from visualization_lib_br import VisualizationLib
from kinematics_lib_br import KinematicsLib
from mesh_depth_lib_br import MeshDepthLib

from utils import print_project_details, log_message
import inspect

class CNN(nn.Module):
    def __init__(self, out_size, loss_vector_type, batch_size, verts_list, in_channels = 3):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''
        log_message("2.2.1", f"{self.__class__.__name__}.{inspect.stack()[0][3]}", start=True)
        print_project_details()
        print(f"_" * 80)
        print(f"\033[1m{'Inputs to CNN':^80}\033[0m")
        print(f"out_size:           {out_size}")
        print(f"loss_vector_type:   {loss_vector_type}")
        print(f"batch_size:         {batch_size}")
        print(f"verts_list:         {verts_list}")
        print(f"in_channels:        {in_channels}")
        print(f"_" * 80)

        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size
        self.loss_vector_type = loss_vector_type

        self.count = 0


        self.CNN_pack1 = nn.Sequential(

            nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )

        self.CNN_packtanh = nn.Sequential(

            nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )

        self.CNN_packtanh_double = nn.Sequential(

            nn.Conv2d(in_channels, 384, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )

        self.CNN_pack2 = nn.Sequential(

            nn.Conv2d(in_channels, 32, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),

        )


        self.CNN_fc1 = nn.Sequential(
            nn.Linear(67200, out_size), #89600, out_size),
        )
        self.CNN_fc1_double = nn.Sequential(
            nn.Linear(67200*2, out_size), #89600, out_size),
        )


        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
        self.dtype = dtype

        self.verts_list = verts_list
        # self.meshDepthLib = MeshDepthLib(loss_vector_type, batch_size, verts_list = self.verts_list)
        # not needed here, it's only needed in the PhysicalTrainer.train_convnet -> UnpackBatchLib().unpack_batch -> model.CNN.forward_kinematic_angles

        log_message("2.2.1", f"{self.__class__.__name__}.{inspect.stack()[0][3]}", start=False)


    def forward_kinematic_angles(self, x_images, y_true_gender_switch, y_true_synth_real_switch, CTRL_PNL, OUTPUT_EST_DICT,
                                 y_true_markers_xyz=None, is_training = True, y_true_betas=None, y_true_angles = None, y_true_root_xyz = None):
        log_message("2.3.1", f"{self.__class__.__name__}.{inspect.stack()[0][3]} (F O R W A R D   P A S S)", start=True)
        print_project_details()

        print(f"_" * 80)
        print(f"\033[1m{'Inputs to forward_kinematic_angles':^80}\033[0m")
        print(f"x_images:                   {x_images.size()}")
        print(f"y_true_markers_xyz:         {y_true_markers_xyz.size()}")
        print(f"y_true_betas:               {y_true_betas.size()}")
        print(f"y_true_angles:              {y_true_angles.size()}")
        print(f"y_true_root_xyz:            {y_true_root_xyz.size()}")
        print(f"y_true_gender_switch:       {y_true_gender_switch.size()}")
        print(f"y_true_synth_real_switch:   {y_true_synth_real_switch.size()}")
        print("_" * 80)

        #cut out the sobel and contact channels
        print(f"CTRL_PNL['omit_cntct_sobel'] == {CTRL_PNL['omit_cntct_sobel']}")
        if CTRL_PNL['omit_cntct_sobel'] == True:

            if CTRL_PNL['cal_noise'] == True:
                x_images = torch.cat((x_images[:, 1:CTRL_PNL['num_input_channels_batch0'], :, :], x_images[:, CTRL_PNL['num_input_channels_batch0']+1:, :, :]), dim = 1)
            else:
                x_images = torch.cat((x_images[:, 1:CTRL_PNL['num_input_channels_batch0']-1, :, :], x_images[:, CTRL_PNL['num_input_channels_batch0']:, :, :]), dim = 1)



        reg_angles = CTRL_PNL['regr_angles']

        OUTPUT_DICT = {}

        self.GPU = CTRL_PNL['GPU']
        self.dtype = CTRL_PNL['dtype']

        #print(torch.cuda.max_memory_allocated(), 'conv0', images.size())
        print(f"CTRL_PNL['first_pass'] == {CTRL_PNL['first_pass']}")
        if CTRL_PNL['first_pass'] == False:
            x = self.SMPL_meshDepthLib.bounds
            #print blah
            #self.GPU = False
            #self.dtype = torch.FloatTensor

        else:
            if CTRL_PNL['GPU'] == True:
                self.GPU = True
                self.dtype = torch.cuda.FloatTensor
            else:
                self.GPU = False
                self.dtype = torch.FloatTensor
            if CTRL_PNL['depth_map_output'] == True:
                self.verts_list = "all"
            else:
                self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
            self.SMPL_meshDepthLib = MeshDepthLib(loss_vector_type=self.loss_vector_type,
                                             batch_size=x_images.size(0), verts_list = self.verts_list)

        print(f"CTRL_PNL['all_tanh_activ'] == {CTRL_PNL['all_tanh_activ']}")
        if CTRL_PNL['all_tanh_activ'] == True:
            print(f"CTRL_PNL['double_network_size'] == {CTRL_PNL['double_network_size']}")
            if CTRL_PNL['double_network_size'] == False:
                print(f"images: {x_images.size()}")
                scores_cnn = self.CNN_packtanh(x_images)
                print(f"scores_cnn: {scores_cnn.size()}")
            else:
                scores_cnn = self.CNN_packtanh_double(x_images)

        else:
            scores_cnn = self.CNN_pack1(x_images)

        scores_size = scores_cnn.size()

        print(f"scores_size: {scores_size}")

        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(x_images.size(0),scores_size[1] *scores_size[2]*scores_size[3])

        print(f"scores_cnn: {scores_cnn.size()}")
        # this output is N x 85: betas, root shift, angles
        print(f"CTRL_PNL['double_network_size'] == {CTRL_PNL['double_network_size']}")
        if CTRL_PNL['double_network_size'] == False:
            y_pred_cnn = self.CNN_fc1(scores_cnn)
        else:
            y_pred_cnn = self.CNN_fc1_double(scores_cnn)

        print(f"scores: {scores.size()}")

        # weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large.
        if CTRL_PNL['adjust_ang_from_est'] == True:
            y_pred_cnn = torch.mul(y_pred_cnn.clone(), 0.01)
        else:
            y_pred_cnn = torch.mul(y_pred_cnn.clone(), 0.01)
        print(f"y_pred_cnn: {y_pred_cnn.size()}")

        #normalize the output of the network based on the range of the parameters
        #if self.GPU == True:
        #    output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + 6*[2.0] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[3:, 1] - self.meshDepthLib.bounds.view(72,2)[3:, 0]).cpu().numpy())
        #else:
        #    output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + 6*[2.0] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[3:, 1] - self.meshDepthLib.bounds.view(72, 2)[3:, 0]).numpy())
        #for i in range(88):
        #    scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])


        #add a factor so the model starts close to the home position. Has nothing to do with weighting.

        print(f"CTRL_PNL['lock_root'] == {CTRL_PNL['lock_root']}")
        print(f"CTRL_PNL['adjust_ang_from_est'] == {CTRL_PNL['adjust_ang_from_est']}")
        if CTRL_PNL['lock_root'] == True:
            y_pred_cnn[:, 10] = torch.add(y_pred_cnn[:, 10].clone(), 0.6).detach()
            y_pred_cnn[:, 11] = torch.add(y_pred_cnn[:, 11].clone(), 1.2).detach()
            y_pred_cnn[:, 12] = torch.add(y_pred_cnn[:, 12].clone(), 0.1).detach()
        elif CTRL_PNL['adjust_ang_from_est'] == True:
            pass
        else:
            y_pred_cnn[:, 10] = torch.add(y_pred_cnn[:, 10].clone(), 0.6)
            y_pred_cnn[:, 11] = torch.add(y_pred_cnn[:, 11].clone(), 1.2)
            y_pred_cnn[:, 12] = torch.add(y_pred_cnn[:, 12].clone(), 0.1)

        #scores[:, 12] = torch.add(scores[:, 12].clone(), 0.06)

        print(f"CTRL_PNL['full_body_rot'] == {CTRL_PNL['full_body_rot']}")
        if CTRL_PNL['full_body_rot'] == True:

            y_pred_cnn = y_pred_cnn.unsqueeze(0)
            print(f"y_pred_cnn.unsqueeze(0): {y_pred_cnn.size()}")
            y_pred_cnn = y_pred_cnn.unsqueeze(0)
            print(f"y_pred_cnn.unsqueeze(0): {y_pred_cnn.size()}")
            y_pred_cnn = F.pad(y_pred_cnn, (0, 3, 0, 0))
            print(f"F.pad(y_pred_cnn, (0, 3, 0, 0)): {y_pred_cnn.size()}")
            y_pred_cnn = y_pred_cnn.squeeze(0)
            print(f"y_pred_cnn.squeeze(0): {y_pred_cnn.size()}")
            y_pred_cnn = y_pred_cnn.squeeze(0)
            print(f"y_pred_cnn.squeeze(0): {y_pred_cnn.size()}")

            if CTRL_PNL['adjust_ang_from_est'] == True:

                y_pred_cnn[:, 13:19] = y_pred_cnn[:, 13:19].clone() + OUTPUT_EST_DICT['root_atan2']


            y_pred_cnn[:, 22:91] = y_pred_cnn[:, 19:88].clone()

            y_pred_cnn[:, 19] = torch.atan2(y_pred_cnn[:, 16].clone(), y_pred_cnn[:, 13].clone()) #pitch x, y
            y_pred_cnn[:, 20] = torch.atan2(y_pred_cnn[:, 17].clone(), y_pred_cnn[:, 14].clone()) #roll x, y
            y_pred_cnn[:, 21] = torch.atan2(y_pred_cnn[:, 18].clone(), y_pred_cnn[:, 15].clone()) #yaw x, y

            OSA = 6 #output size adder
            print(f"OSA: {OSA}")
        else:
            OSA = 0




        #print scores[0, 0:10]
        if CTRL_PNL['adjust_ang_from_est'] == True: # in our case, False
            y_pred_cnn[:, 0:10] =  OUTPUT_EST_DICT['betas'] + y_pred_cnn[:, 0:10].clone()
            y_pred_cnn[:, 10:13] = OUTPUT_EST_DICT['root_shift'] + y_pred_cnn[:, 10:13].clone()
            if CTRL_PNL['full_body_rot'] == True:
                y_pred_cnn[:, 22:91] = y_pred_cnn[:, 22:91].clone() + OUTPUT_EST_DICT['angles'][:, 3:72]
            else:
                y_pred_cnn[:, 13:85] = y_pred_cnn[:, 13:85].clone() + OUTPUT_EST_DICT['angles']
            #scores[:, 13:85] = OUTPUT_EST_DICT['angles']


        OUTPUT_DICT['y_pred_betas']     = y_pred_cnn[:, 0:10].clone().data
        if CTRL_PNL['full_body_rot'] == True:
            OUTPUT_DICT['y_pred_root_atan2'] = y_pred_cnn[:, 13:19].clone().data
        OUTPUT_DICT['y_pred_angles']    = y_pred_cnn[:, 13+OSA:85+OSA].clone().data
        OUTPUT_DICT['y_pred_root_xyz']  = y_pred_cnn[:, 10:13].clone().data


        print(f"reg_angles == {reg_angles}") # in our case, False
        if reg_angles == True:
            add_idx = 72
        else:
            add_idx = 0
        print(f"add_idx: {add_idx}")


        print(f"CTRL_PNL['clip_betas'] == {CTRL_PNL['clip_betas']}") # in our case, True
        if CTRL_PNL['clip_betas'] == True:
            y_pred_cnn[:, 0:10] /= 3.
            y_pred_cnn[:, 0:10] = y_pred_cnn[:, 0:10].tanh()
            y_pred_cnn[:, 0:10] *= 3.

        #print self.meshDepthLib.bounds

        test_ground_truth = False #can only use True when the dataset is entirely synthetic AND when we use anglesDC
        #is_training = True

        if test_ground_truth == False or is_training == False:
            # make sure the estimated betas are reasonable.

            y_pred_betas    = y_pred_cnn[:, 0:10].clone()#.detach() #make sure to detach so the gradient flow of joints doesn't corrupt the betas
            y_pred_root_xyz = y_pred_cnn[:, 10:13].clone()




            # normalize for tan activation function
            y_pred_cnn[:, 13+OSA:85+OSA] -= torch.mean(self.SMPL_meshDepthLib.bounds[0:72, 0:2], dim=1)
            y_pred_cnn[:, 13+OSA:85+OSA] *= (2. / torch.abs(self.SMPL_meshDepthLib.bounds[0:72, 0] - self.SMPL_meshDepthLib.bounds[0:72, 1]))
            y_pred_cnn[:, 13+OSA:85+OSA] = y_pred_cnn[:, 13+OSA:85+OSA].tanh()
            y_pred_cnn[:, 13+OSA:85+OSA] /= (2. / torch.abs(self.SMPL_meshDepthLib.bounds[0:72, 0] - self.SMPL_meshDepthLib.bounds[0:72, 1]))
            y_pred_cnn[:, 13+OSA:85+OSA] += torch.mean(self.SMPL_meshDepthLib.bounds[0:72, 0:2], dim=1)


            CTRL_PNL['align_procr'] = False
            if CTRL_PNL['align_procr'] == True:
                print("aligning procrustes")
                y_pred_root_xyz = y_true_root_xyz
                y_pred_cnn[:, 13+OSA:16+OSA] = y_true_angles[:, 0:3].clone()


            #print scores[:, 13+OSA:85+OSA]


            if self.loss_vector_type == 'anglesDC':

                y_pred_angles_rot_mat = KinematicsLib().batch_rodrigues(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

                print(f"Rs_est: {Rs_est.size()}")
            elif self.loss_vector_type == 'anglesEU':

                y_pred_angles_rot_mat = KinematicsLib().batch_euler_to_R(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.SMPL_meshDepthLib.zeros_cartesian, self.SMPL_meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            y_pred_betas = y_true_betas
            y_pred_cnn[:, 0:10] = y_true_betas.clone()
            y_pred_cnn[:, 13+OSA:85+OSA] = y_true_angles.clone()
            y_pred_root_xyz = y_true_root_xyz


            if self.loss_vector_type == 'anglesDC':

                #normalize for tan activation function
                #scores[:, 13+OSA:85+OSA] -= torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)
                #scores[:, 13+OSA:85+OSA] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] = scores[:, 13+OSA:85+OSA].tanh()
                #scores[:, 13+OSA:85+OSA] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] += torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)


                y_pred_angles_rot_mat = KinematicsLib().batch_rodrigues(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

        #print Rs_est[0, :]


        OUTPUT_DICT['y_pred_betas_post_clip']       = y_pred_cnn[:, 0:10].clone().data
        if self.loss_vector_type == 'anglesEU':
            OUTPUT_DICT['y_pred_angles_post_clip']  = KinematicsLib().batch_dir_cos_angles_from_euler_angles(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.SMPL_meshDepthLib.zeros_cartesian, self.SMPL_meshDepthLib.ones_cartesian)
        elif self.loss_vector_type == 'anglesDC':
            OUTPUT_DICT['y_pred_angles_post_clip']  = y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()
        OUTPUT_DICT['y_pred_root_xyz_post_clip']    = y_pred_cnn[:, 10:13].clone().data


        y_true_gender_switch = y_true_gender_switch.unsqueeze(1)
        current_batch_size = y_true_gender_switch.size()[0]


        print(f"CTRL_PNL['depth_map_output'] == {CTRL_PNL['depth_map_output']}")
        print(f"y_true_gender_switch.unsqueeze(1): {y_true_gender_switch.size()}")
        print(f"current_batch_size:         {current_batch_size}")
        if CTRL_PNL['depth_map_output'] == True:
            # break things up into sub batches and pass through the mesh
            num_normal_sub_batches = current_batch_size // self.SMPL_meshDepthLib.batch_size
            if current_batch_size % self.SMPL_meshDepthLib.batch_size != 0:
                sub_batch_incr_list = num_normal_sub_batches * [self.SMPL_meshDepthLib.batch_size] + [current_batch_size % self.SMPL_meshDepthLib.batch_size]
            else:
                sub_batch_incr_list = num_normal_sub_batches * [self.SMPL_meshDepthLib.batch_size]
            start_incr, end_incr = 0, 0

            #print len(sub_batch_incr_list), current_batch_size

            for sub_batch_incr in sub_batch_incr_list:
                end_incr += sub_batch_incr
                verts_sub, J_est_sub, targets_est_sub = self.SMPL_meshDepthLib.HMR(y_true_gender_switch, y_pred_betas,
                                                                              y_pred_angles_rot_mat, y_pred_root_xyz,
                                                                              start_incr, end_incr, self.GPU)
                if start_incr == 0:
                    SMPL_pred_verts = verts_sub.clone()
                    SMPL_pred_J = J_est_sub.clone()
                    y_pred_markers_xyz = targets_est_sub.clone()
                else:
                    SMPL_pred_verts = torch.cat((SMPL_pred_verts, verts_sub), dim=0)
                    SMPL_pred_J = torch.cat((SMPL_pred_J, J_est_sub), dim=0)
                    y_pred_markers_xyz = torch.cat((y_pred_markers_xyz, targets_est_sub), dim=0)
                start_incr += sub_batch_incr

            bed_ang_idx = -1
            if CTRL_PNL['incl_ht_wt_channels'] == True: bed_ang_idx -= 2
            bed_angle_batch = torch.mean(x_images[:, bed_ang_idx, 1:3, 0], dim=1)

            OUTPUT_DICT['batch_mdm_est'], OUTPUT_DICT['batch_cm_est'] = self.SMPL_meshDepthLib.PMR(SMPL_pred_verts, bed_angle_batch,
                                                                                            CTRL_PNL['mesh_bottom_dist'])

            OUTPUT_DICT['batch_mdm_est'] = OUTPUT_DICT['batch_mdm_est'].type(self.dtype)
            OUTPUT_DICT['batch_cm_est'] = OUTPUT_DICT['batch_cm_est'].type(self.dtype)

            verts_red = torch.stack([SMPL_pred_verts[:, 1325, :],
                                     SMPL_pred_verts[:, 336, :],  # head
                                     SMPL_pred_verts[:, 1032, :],  # l knee
                                     SMPL_pred_verts[:, 4515, :],  # r knee
                                     SMPL_pred_verts[:, 1374, :],  # l ankle
                                     SMPL_pred_verts[:, 4848, :],  # r ankle
                                     SMPL_pred_verts[:, 1739, :],  # l elbow
                                     SMPL_pred_verts[:, 5209, :],  # r elbow
                                     SMPL_pred_verts[:, 1960, :],  # l wrist
                                     SMPL_pred_verts[:, 5423, :]]).permute(1, 0, 2)  # r wrist

            SMPL_pred_verts_offset = verts_red.clone().detach().cpu()
            SMPL_pred_verts_offset = torch.Tensor(SMPL_pred_verts_offset.numpy()).type(self.dtype)

        else:
            print("_" * 50)
            print(f"SMPL_shapedirs = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_shapedirs_repeat[0:current_batch_size, :, :]).view(current_batch_size, SMPL_meshDepthLib.SMPL_B, SMPL_meshDepthLib.SMPL_R * SMPL_meshDepthLib.SMPL_D)")
            SMPL_shapedirs = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_shapedirs_repeat[0:current_batch_size, :, :])\
                             .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_B, self.SMPL_meshDepthLib.SMPL_R*self.SMPL_meshDepthLib.SMPL_D)
            print(f"y_true_gender_switch:  {y_true_gender_switch.size()}")
            print(f"SMPL_meshDepthLib.SMPL_shapedirs_repeat: {self.SMPL_meshDepthLib.SMPL_shapedirs_repeat[0:current_batch_size, :, :].size()}")
            print(f"torch.bmm():    {torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_shapedirs_repeat[0:current_batch_size, :, :]).size()}")
            print(f"SMPL_shapedirs:      {SMPL_shapedirs.size()}")
            print("_" * 50)

            print("_" * 50)
            print(f"betas_shapedirs_mult = torch.bmm(betas.unsqueeze(1), shapedirs).squeeze(1).view(current_batch_size, R, D)")
            SMPL_shapedirs_y_pred_betas_mult = torch.bmm(y_pred_betas.unsqueeze(1), SMPL_shapedirs)\
                                        .squeeze(1)\
                                        .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, self.SMPL_meshDepthLib.SMPL_D)
            print(f"y_pred_betas.unsqueeze(1): {y_pred_betas.unsqueeze(1).size()}")
            print(f"SMPL_shapedirs:              {SMPL_shapedirs.size()}")
            print(f"torch.bmm():            {torch.bmm(y_pred_betas.unsqueeze(1), SMPL_shapedirs).size()}")
            print(f"torch.bmm().squeeze(1): {torch.bmm(y_pred_betas.unsqueeze(1), SMPL_shapedirs).squeeze(1).size()}")
            print(f"SMPL_shapedirs_y_pred_betas_mult:   {SMPL_shapedirs_y_pred_betas_mult.size()}")

            SMPL_v_template = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_v_template_repeat[0:current_batch_size, :, :])\
                              .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, self.SMPL_meshDepthLib.SMPL_D)
            print(f"SMPL_v_template: {SMPL_v_template.size()}")

            SMPL_pred_v_shaped = SMPL_shapedirs_y_pred_betas_mult + SMPL_v_template
            print(f"SMPL_pred_v_shaped: {SMPL_pred_v_shaped.size()}")

            SMPL_J_regressor = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_J_regressor_repeat[0:current_batch_size, :, :])\
                                      .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, 24)
            print(f"SMPL_J_regressor: {SMPL_J_regressor.size()}")

            SMPL_pred_Jx = torch.bmm(SMPL_pred_v_shaped[:, :, 0].unsqueeze(1), SMPL_J_regressor).squeeze(1)
            SMPL_pred_Jy = torch.bmm(SMPL_pred_v_shaped[:, :, 1].unsqueeze(1), SMPL_J_regressor).squeeze(1)
            SMPL_pred_Jz = torch.bmm(SMPL_pred_v_shaped[:, :, 2].unsqueeze(1), SMPL_J_regressor).squeeze(1)
            print(f"SMPL_pred_Jx: {SMPL_pred_Jx.size()}")
            print(f"SMPL_pred_Jy: {SMPL_pred_Jy.size()}")
            print(f"SMPL_pred_Jz: {SMPL_pred_Jz.size()}")


            SMPL_pred_J = torch.stack([SMPL_pred_Jx, SMPL_pred_Jy, SMPL_pred_Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
            #J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)
            print(f"SMPL_pred_J: {SMPL_pred_J.size()}")


            y_pred_markers_xyz, SMPL_pred_A = KinematicsLib().batch_global_rigid_transformation(y_pred_angles_rot_mat, SMPL_pred_J, self.SMPL_meshDepthLib.parents,
                                                                                   self.GPU, rotate_base=False)
            print(f"y_pred_markers_xyz: {y_pred_markers_xyz.size()}")
            print(f"SMPL_pred_A: {SMPL_pred_A.size()}")

            y_pred_markers_xyz = y_pred_markers_xyz - SMPL_pred_J[:, 0:1, :] + y_pred_root_xyz.unsqueeze(1)
            print(f"y_pred_markers_xyz: {y_pred_markers_xyz.size()}")

            # assemble a reduced form of the transformed mesh
            SMPL_pred_v_shaped_red = torch.stack([SMPL_pred_v_shaped[:, self.verts_list[0], :],
                                        SMPL_pred_v_shaped[:, self.verts_list[1], :],  # head
                                        SMPL_pred_v_shaped[:, self.verts_list[2], :],  # l knee
                                        SMPL_pred_v_shaped[:, self.verts_list[3], :],  # r knee
                                        SMPL_pred_v_shaped[:, self.verts_list[4], :],  # l ankle
                                        SMPL_pred_v_shaped[:, self.verts_list[5], :],  # r ankle
                                        SMPL_pred_v_shaped[:, self.verts_list[6], :],  # l elbow
                                        SMPL_pred_v_shaped[:, self.verts_list[7], :],  # r elbow
                                        SMPL_pred_v_shaped[:, self.verts_list[8], :],  # l wrist
                                        SMPL_pred_v_shaped[:, self.verts_list[9], :]]).permute(1, 0, 2)  # r wrist
            print(f"SMPL_pred_v_shaped_red:           {SMPL_pred_v_shaped_red.size()}")

            # y_pred_angles_rot_mat_pose_feature = (y_pred_angles_rot_mat[:, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)   # replaced by Nadeem, as .sub is deprecated
            y_pred_angles_rot_mat_pose_feature = (y_pred_angles_rot_mat[:, 1:, :, :] - torch.eye(3).type(self.dtype)).view(-1, 207)
            # y_pred_angles_rot_mat_pose_feature = (y_pred_angles_rot_mat[:, 1:, :, :] - torch.eye(3).type(self.dtype)).view(-1, 207)
                            #  ([512, 23, 3, 3] - [3, 3]).view(-1, 93 or 207)

            print(f"y_pred_angles_rot_mat_pose_feature:           {y_pred_angles_rot_mat_pose_feature.size()}")

            SMPL_posedirs = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_posedirs_repeat[0:current_batch_size, :, :]) \
                .view(current_batch_size, 10 * self.SMPL_meshDepthLib.SMPL_D, 207) \
                .permute(0, 2, 1)
            print(f"SMPL_posedirs: {SMPL_posedirs.size()}")

            SMPL_pred_v_posed = torch.bmm(y_pred_angles_rot_mat_pose_feature.unsqueeze(1), SMPL_posedirs).view(-1, 10, self.SMPL_meshDepthLib.SMPL_D)
            print(f"SMPL_pred_v_posed: {SMPL_pred_v_posed.size()}")

            SMPL_pred_v_posed = SMPL_pred_v_posed.clone() + SMPL_pred_v_shaped_red
            SMPL_weights = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_weights_repeat[0:current_batch_size, :, :]) \
                .squeeze(1) \
                .view(current_batch_size, 10, 24)
            SMPL_pred_T = torch.bmm(SMPL_weights, SMPL_pred_A.view(current_batch_size, 24, 16)).view(current_batch_size, -1, 4, 4)
            SMPL_pred_v_posed_homo = torch.cat([SMPL_pred_v_posed, torch.ones(current_batch_size, SMPL_pred_v_posed.shape[1], 1).type(self.dtype)], dim=2)
            SMPL_pred_v_homo = torch.matmul(SMPL_pred_T, torch.unsqueeze(SMPL_pred_v_posed_homo, -1))


            SMPL_pred_verts = SMPL_pred_v_homo[:, :, :3, 0] - SMPL_pred_J[:, 0:1, :] + y_pred_root_xyz.unsqueeze(1)

            SMPL_pred_verts_offset = torch.Tensor(SMPL_pred_verts.clone().detach().cpu().numpy()).type(self.dtype)

            OUTPUT_DICT['batch_mdm_est'] = None
            OUTPUT_DICT['batch_cm_est'] = None


# ---------------------------------------------------------------------------------------------------
        #print verts[0:10], 'VERTS EST INIT'
        OUTPUT_DICT['SMPL_pred_verts'] = SMPL_pred_verts.clone().detach().cpu().numpy()

        SMPL_pred_targets_detached = torch.Tensor(y_pred_markers_xyz.clone().detach().cpu().numpy()).type(self.dtype)
        synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]
        for real_joint in range(10):
            SMPL_pred_verts_offset[:, real_joint, :] = SMPL_pred_verts_offset[:, real_joint, :] - SMPL_pred_targets_detached[:, synth_joint_addressed[real_joint], :]


        #here we need to the ground truth to make it a surface point for the mocap markers
        #if is_training == True:
        y_true_synth_real_switch_repeated = y_true_synth_real_switch.unsqueeze(1).repeat(1, 3)
        for real_joint in range(10):
            y_pred_markers_xyz[:, synth_joint_addressed[real_joint], :] = y_true_synth_real_switch_repeated * y_pred_markers_xyz[:, synth_joint_addressed[real_joint], :].clone() \
                                   + torch.add(-y_true_synth_real_switch_repeated, 1) * (y_pred_markers_xyz[:, synth_joint_addressed[real_joint], :].clone() + SMPL_pred_verts_offset[:, real_joint, :])


        y_pred_markers_xyz = y_pred_markers_xyz.contiguous().view(-1, 72)

        OUTPUT_DICT['y_pred_markers_xyz'] = y_pred_markers_xyz.data*1000. #after it comes out of the forward kinematics

        y_pred_cnn = y_pred_cnn.unsqueeze(0)
        y_pred_cnn = y_pred_cnn.unsqueeze(0)
        y_pred_cnn = F.pad(y_pred_cnn, (0, 100 + add_idx, 0, 0))
        y_pred_cnn = y_pred_cnn.squeeze(0)
        y_pred_cnn = y_pred_cnn.squeeze(0)


        #tweak this to change the lengths vector
        y_pred_cnn[:, 34+add_idx+OSA:106+add_idx+OSA] = torch.mul(y_pred_markers_xyz[:, 0:72], 1.)

        y_pred_cnn[:, 0:10] = torch.mul(y_true_synth_real_switch.unsqueeze(1), torch.sub(y_pred_cnn[:, 0:10], y_true_betas))#*.2
        if CTRL_PNL['full_body_rot'] == True:
            y_pred_cnn[:, 10:16] = y_pred_cnn[:, 13:19].clone()
            if self.loss_vector_type == 'anglesEU':
                y_pred_cnn[:, 10:13] = y_pred_cnn[:, 10:13].clone() - torch.cos(KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
                y_pred_cnn[:, 13:16] = y_pred_cnn[:, 13:16].clone() - torch.sin(KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
            elif self.loss_vector_type == 'anglesDC':
                y_pred_cnn[:, 10:13] = y_pred_cnn[:, 10:13].clone() - torch.cos(y_true_angles[:, 0:3].clone())
                y_pred_cnn[:, 13:16] = y_pred_cnn[:, 13:16].clone() - torch.sin(y_true_angles[:, 0:3].clone())

            #print euler_root_rot_gt[0, :], 'body rot angles gt'

        #compare the output angles to the target values
        if reg_angles == True:
            if self.loss_vector_type == 'anglesDC':
                y_pred_cnn[:, 34+OSA:106+OSA] = y_true_angles.clone().view(-1, 72) - y_pred_cnn[:, 13+OSA:85+OSA]
                y_pred_cnn[:, 34+OSA:106+OSA] = torch.mul(y_true_synth_real_switch.unsqueeze(1), torch.sub(y_pred_cnn[:, 34+OSA:106+OSA], y_true_angles.clone().view(-1, 72)))

            elif self.loss_vector_type == 'anglesEU':
                y_pred_cnn[:, 34+OSA:106+OSA] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles.view(-1, 24, 3).clone()).contiguous().view(-1, 72) - y_pred_cnn[:, 13+OSA:85+OSA]

            y_pred_cnn[:, 34+OSA:106+OSA] = torch.mul(y_true_synth_real_switch.unsqueeze(1), y_pred_cnn[:, 34+OSA:106+OSA].clone())



        #compare the output joints to the target values

        y_pred_cnn[:, 34+add_idx+OSA:106+add_idx+OSA] = y_true_markers_xyz[:, 0:72]/1000 - y_pred_cnn[:, 34+add_idx+OSA:106+add_idx+OSA]
        y_pred_cnn[:, 106+add_idx+OSA:178+add_idx+OSA] = ((y_pred_cnn[:, 34+add_idx+OSA:106+add_idx+OSA].clone())+0.0000001).pow(2)


        for joint_num in range(24):
            if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                y_pred_cnn[:, 10+joint_num+OSA] = torch.mul(y_true_synth_real_switch,
                                                    (y_pred_cnn[:, 106+add_idx+joint_num*3+OSA] +
                                                     y_pred_cnn[:, 107+add_idx+joint_num*3+OSA] +
                                                     y_pred_cnn[:, 108+add_idx+joint_num*3+OSA]).sqrt())

            else:
                y_pred_cnn[:, 10+joint_num+OSA] = (y_pred_cnn[:, 106+add_idx+joint_num*3+OSA] +
                                           y_pred_cnn[:, 107+add_idx+joint_num*3+OSA] +
                                           y_pred_cnn[:, 108+add_idx+joint_num*3+OSA]).sqrt()


        y_pred_cnn = y_pred_cnn.unsqueeze(0)
        y_pred_cnn = y_pred_cnn.unsqueeze(0)
        y_pred_cnn = F.pad(y_pred_cnn, (0, -151, 0, 0))
        y_pred_cnn = y_pred_cnn.squeeze(0)
        y_pred_cnn = y_pred_cnn.squeeze(0)


        y_pred_cnn[:, 0:10] = torch.mul(y_pred_cnn[:, 0:10].clone(), (1/1.728158146914805))#1.7312621950698526)) #weight the betas by std
        if CTRL_PNL['full_body_rot'] == True:
            y_pred_cnn[:, 10:16] = torch.mul(y_pred_cnn[:, 10:16].clone(), (1/0.3684988513298487))#0.2130542427733348)*np.pi) #weight the body rotation by the std
        y_pred_cnn[:, 10+OSA:34+OSA] = torch.mul(y_pred_cnn[:, 10+OSA:34+OSA].clone(), (1/0.1752780723422608))#0.1282715100608753)) #weight the 24 joints by std
        if reg_angles == True: y_pred_cnn[:, 34+OSA:106+OSA] = torch.mul(y_pred_cnn[:, 34+OSA:106+OSA].clone(), (1/0.29641429463719227))#0.2130542427733348)) #weight the angles by how many there are

        #scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./10)) #weight the betas by how many betas there are
        #scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), (1./24)) #weight the joints by how many there are
        #if reg_angles == True: scores[:, 34:106] = torch.mul(scores[:, 34:106].clone(), (1./72)) #weight the angles by how many there are


        return y_pred_cnn, OUTPUT_DICT
        log_message("2.3.1", f"{self.__class__.__name__}.{inspect.stack()[0][3]} (F O R W A R D   P A S S)", start=False)
        return scores, OUTPUT_DICT

