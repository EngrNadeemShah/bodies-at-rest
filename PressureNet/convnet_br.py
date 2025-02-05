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


from visualization_lib_br import VisualizationLib
from kinematics_lib_br import KinematicsLib
from mesh_depth_lib_br import MeshDepthLib


class CNN(nn.Module):
    def __init__(self, convnet_fc_output_size, loss_type, vertices, in_channels = 3):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
        convnet_fc_output_size: int, number of output units in the fully connected layer
        loss_type: string, type of loss function to use
        vertices: list of ints, indices of the vertices to use in the mesh
        in_channels: int, number of channels in the input images
        '''

        super(CNN, self).__init__()

        self.loss_type = loss_type
        self.vertices = vertices
        self.count = 0

        self.GPU = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.GPU else torch.FloatTensor

        # # Determine if CUDA is available and set the device accordingly
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


        self.CNN_fc1 = nn.Sequential(nn.Linear(67200, convnet_fc_output_size))          # 89600, convnet_fc_output_size
        self.CNN_fc1_double = nn.Sequential(nn.Linear(67200*2, convnet_fc_output_size)) # 89600, convnet_fc_output_size


    def forward_kinematic_angles(self, x_images, y_true_gender_switch, y_true_synth_real_switch, CTRL_PNL, OUTPUT_EST_DICT,
                                 y_true_markers_xyz=None, is_training = True, y_true_betas=None, y_true_angles = None, y_true_root_xyz = None):

        # Cut out the sobel and contact channels
        if CTRL_PNL['omit_cntct_sobel'] == True:

            if CTRL_PNL['cal_noise'] == True:
                x_images = torch.cat((x_images[:, 1:CTRL_PNL['num_input_channels_batch0'], :, :], x_images[:, CTRL_PNL['num_input_channels_batch0']+1:, :, :]), dim = 1)
            else:
                x_images = torch.cat((x_images[:, 1:CTRL_PNL['num_input_channels_batch0']-1, :, :], x_images[:, CTRL_PNL['num_input_channels_batch0']:, :, :]), dim = 1)



        reg_angles = CTRL_PNL['regr_angles']

        OUTPUT_DICT = {}

        self.GPU = CTRL_PNL['GPU']
        self.dtype = CTRL_PNL['dtype']

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
                self.vertices = "all"
            else:
                self.vertices = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
            self.SMPL_meshDepthLib = MeshDepthLib(loss_type=self.loss_type, batch_size=x_images.size(0), verts_list = self.vertices)

        if CTRL_PNL['all_tanh_activ'] == True:
            if CTRL_PNL['double_network_size'] == False:
                scores_cnn = self.CNN_packtanh(x_images)
            else:
                scores_cnn = self.CNN_packtanh_double(x_images)

        else:
            scores_cnn = self.CNN_pack1(x_images)

        scores_size = scores_cnn.size()


        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(x_images.size(0),scores_size[1] *scores_size[2]*scores_size[3])

        # This output is N x 85: betas, root shift, angles
        if CTRL_PNL['double_network_size'] == False:
            y_pred_cnn = self.CNN_fc1(scores_cnn)
        else:
            y_pred_cnn = self.CNN_fc1_double(scores_cnn)


        # weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large.
        if CTRL_PNL['adjust_ang_from_est'] == True:
            y_pred_cnn = torch.mul(y_pred_cnn.clone(), 0.01)
        else:
            y_pred_cnn = torch.mul(y_pred_cnn.clone(), 0.01)


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


        if CTRL_PNL['full_body_rot'] == True:

            y_pred_cnn = y_pred_cnn.unsqueeze(0)
            y_pred_cnn = y_pred_cnn.unsqueeze(0)
            y_pred_cnn = F.pad(y_pred_cnn, (0, 3, 0, 0))
            y_pred_cnn = y_pred_cnn.squeeze(0)
            y_pred_cnn = y_pred_cnn.squeeze(0)

            if CTRL_PNL['adjust_ang_from_est'] == True:

                y_pred_cnn[:, 13:19] = y_pred_cnn[:, 13:19].clone() + OUTPUT_EST_DICT['root_atan2']


            y_pred_cnn[:, 22:91] = y_pred_cnn[:, 19:88].clone()

            y_pred_cnn[:, 19] = torch.atan2(y_pred_cnn[:, 16].clone(), y_pred_cnn[:, 13].clone()) #pitch x, y
            y_pred_cnn[:, 20] = torch.atan2(y_pred_cnn[:, 17].clone(), y_pred_cnn[:, 14].clone()) #roll x, y
            y_pred_cnn[:, 21] = torch.atan2(y_pred_cnn[:, 18].clone(), y_pred_cnn[:, 15].clone()) #yaw x, y

            OSA = 6 #output size adder
        else:
            OSA = 0




        if CTRL_PNL['adjust_ang_from_est'] == True:
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


        if reg_angles == True:
            add_idx = 72
        else:
            add_idx = 0


        if CTRL_PNL['clip_betas'] == True:
            y_pred_cnn[:, 0:10] /= 3.
            y_pred_cnn[:, 0:10] = y_pred_cnn[:, 0:10].tanh()
            y_pred_cnn[:, 0:10] *= 3.


        test_ground_truth = False # can only use True when the dataset is entirely synthetic AND when we use anglesDC

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


            if self.loss_type == 'anglesDC':

                y_pred_angles_rot_mat = KinematicsLib().batch_rodrigues(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            elif self.loss_type == 'anglesEU':

                y_pred_angles_rot_mat = KinematicsLib().batch_euler_to_R(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.SMPL_meshDepthLib.zeros_cartesian, self.SMPL_meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            y_pred_betas = y_true_betas
            y_pred_cnn[:, 0:10] = y_true_betas.clone()
            y_pred_cnn[:, 13+OSA:85+OSA] = y_true_angles.clone()
            y_pred_root_xyz = y_true_root_xyz


            if self.loss_type == 'anglesDC':

                #normalize for tan activation function
                #scores[:, 13+OSA:85+OSA] -= torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)
                #scores[:, 13+OSA:85+OSA] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] = scores[:, 13+OSA:85+OSA].tanh()
                #scores[:, 13+OSA:85+OSA] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] += torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)


                y_pred_angles_rot_mat = KinematicsLib().batch_rodrigues(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)


        OUTPUT_DICT['y_pred_betas_post_clip']       = y_pred_cnn[:, 0:10].clone().data
        if self.loss_type == 'anglesEU':
            OUTPUT_DICT['y_pred_angles_post_clip']  = KinematicsLib().batch_dir_cos_angles_from_euler_angles(y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.SMPL_meshDepthLib.zeros_cartesian, self.SMPL_meshDepthLib.ones_cartesian)
        elif self.loss_type == 'anglesDC':
            OUTPUT_DICT['y_pred_angles_post_clip']  = y_pred_cnn[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()
        OUTPUT_DICT['y_pred_root_xyz_post_clip']    = y_pred_cnn[:, 10:13].clone().data


        y_true_gender_switch = y_true_gender_switch.unsqueeze(1)
        current_batch_size = y_true_gender_switch.size()[0]


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
            SMPL_shapedirs = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_shapedirs_repeat[0:current_batch_size, :, :])\
                             .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_B, self.SMPL_meshDepthLib.SMPL_R*self.SMPL_meshDepthLib.SMPL_D)

            SMPL_shapedirs_y_pred_betas_mult = torch.bmm(y_pred_betas.unsqueeze(1), SMPL_shapedirs)\
                                        .squeeze(1)\
                                        .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, self.SMPL_meshDepthLib.SMPL_D)

            SMPL_v_template = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_v_template_repeat[0:current_batch_size, :, :])\
                              .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, self.SMPL_meshDepthLib.SMPL_D)

            SMPL_pred_v_shaped = SMPL_shapedirs_y_pred_betas_mult + SMPL_v_template

            SMPL_J_regressor = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_J_regressor_repeat[0:current_batch_size, :, :])\
                                      .view(current_batch_size, self.SMPL_meshDepthLib.SMPL_R, 24)

            SMPL_pred_Jx = torch.bmm(SMPL_pred_v_shaped[:, :, 0].unsqueeze(1), SMPL_J_regressor).squeeze(1)
            SMPL_pred_Jy = torch.bmm(SMPL_pred_v_shaped[:, :, 1].unsqueeze(1), SMPL_J_regressor).squeeze(1)
            SMPL_pred_Jz = torch.bmm(SMPL_pred_v_shaped[:, :, 2].unsqueeze(1), SMPL_J_regressor).squeeze(1)


            SMPL_pred_J = torch.stack([SMPL_pred_Jx, SMPL_pred_Jy, SMPL_pred_Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
            #J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)


            y_pred_markers_xyz, SMPL_pred_A = KinematicsLib().batch_global_rigid_transformation(y_pred_angles_rot_mat, SMPL_pred_J, self.SMPL_meshDepthLib.parents,
                                                                                   self.GPU, rotate_base=False)

            y_pred_markers_xyz = y_pred_markers_xyz - SMPL_pred_J[:, 0:1, :] + y_pred_root_xyz.unsqueeze(1)

            # assemble a reduced form of the transformed mesh
            SMPL_pred_v_shaped_red = torch.stack([SMPL_pred_v_shaped[:, self.vertices[0], :],
                                        SMPL_pred_v_shaped[:, self.vertices[1], :],  # head
                                        SMPL_pred_v_shaped[:, self.vertices[2], :],  # l knee
                                        SMPL_pred_v_shaped[:, self.vertices[3], :],  # r knee
                                        SMPL_pred_v_shaped[:, self.vertices[4], :],  # l ankle
                                        SMPL_pred_v_shaped[:, self.vertices[5], :],  # r ankle
                                        SMPL_pred_v_shaped[:, self.vertices[6], :],  # l elbow
                                        SMPL_pred_v_shaped[:, self.vertices[7], :],  # r elbow
                                        SMPL_pred_v_shaped[:, self.vertices[8], :],  # l wrist
                                        SMPL_pred_v_shaped[:, self.vertices[9], :]]).permute(1, 0, 2)  # r wrist
            # y_pred_angles_rot_mat_pose_feature = (y_pred_angles_rot_mat[:, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)   # replaced by Nadeem, as .sub is deprecated
            y_pred_angles_rot_mat_pose_feature = (y_pred_angles_rot_mat[:, 1:, :, :] - torch.eye(3).type(self.dtype)).view(-1, 207)
            SMPL_posedirs = torch.bmm(y_true_gender_switch, self.SMPL_meshDepthLib.SMPL_posedirs_repeat[0:current_batch_size, :, :]) \
                .view(current_batch_size, 10 * self.SMPL_meshDepthLib.SMPL_D, 207) \
                .permute(0, 2, 1)
            SMPL_pred_v_posed = torch.bmm(y_pred_angles_rot_mat_pose_feature.unsqueeze(1), SMPL_posedirs).view(-1, 10, self.SMPL_meshDepthLib.SMPL_D)
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


        OUTPUT_DICT['SMPL_pred_verts'] = SMPL_pred_verts.clone().detach().cpu().numpy()

        SMPL_pred_targets_detached = torch.Tensor(y_pred_markers_xyz.clone().detach().cpu().numpy()).type(self.dtype)
        synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]
        for real_joint in range(10):
            SMPL_pred_verts_offset[:, real_joint, :] = SMPL_pred_verts_offset[:, real_joint, :] - SMPL_pred_targets_detached[:, synth_joint_addressed[real_joint], :]


        # here we need to the ground truth to make it a surface point for the mocap markers
        # if is_training == True:
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


        # tweak this to change the lengths vector
        y_pred_cnn[:, 34+add_idx+OSA:106+add_idx+OSA] = torch.mul(y_pred_markers_xyz[:, 0:72], 1.)

        y_pred_cnn[:, 0:10] = torch.mul(y_true_synth_real_switch.unsqueeze(1), torch.sub(y_pred_cnn[:, 0:10], y_true_betas))#*.2
        if CTRL_PNL['full_body_rot'] == True:
            y_pred_cnn[:, 10:16] = y_pred_cnn[:, 13:19].clone()
            if self.loss_type == 'anglesEU':
                y_pred_cnn[:, 10:13] = y_pred_cnn[:, 10:13].clone() - torch.cos(KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
                y_pred_cnn[:, 13:16] = y_pred_cnn[:, 13:16].clone() - torch.sin(KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
            elif self.loss_type == 'anglesDC':
                y_pred_cnn[:, 10:13] = y_pred_cnn[:, 10:13].clone() - torch.cos(y_true_angles[:, 0:3].clone())
                y_pred_cnn[:, 13:16] = y_pred_cnn[:, 13:16].clone() - torch.sin(y_true_angles[:, 0:3].clone())

            #print euler_root_rot_gt[0, :], 'body rot angles gt'

        # compare the output angles to the target values
        if reg_angles == True:
            if self.loss_type == 'anglesDC':
                y_pred_cnn[:, 34+OSA:106+OSA] = y_true_angles.clone().view(-1, 72) - y_pred_cnn[:, 13+OSA:85+OSA]
                y_pred_cnn[:, 34+OSA:106+OSA] = torch.mul(y_true_synth_real_switch.unsqueeze(1), torch.sub(y_pred_cnn[:, 34+OSA:106+OSA], y_true_angles.clone().view(-1, 72)))

            elif self.loss_type == 'anglesEU':
                y_pred_cnn[:, 34+OSA:106+OSA] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(y_true_angles.view(-1, 24, 3).clone()).contiguous().view(-1, 72) - y_pred_cnn[:, 13+OSA:85+OSA]

            y_pred_cnn[:, 34+OSA:106+OSA] = torch.mul(y_true_synth_real_switch.unsqueeze(1), y_pred_cnn[:, 34+OSA:106+OSA].clone())



        # compare the output joints to the target values

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


        return y_pred_cnn, OUTPUT_DICT

