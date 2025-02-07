import sys
import os
import time
import numpy as np
from matplotlib.pylab import *
import torch
import pickle as pickle
import scipy

sys.path.append(os.path.abspath('..'))
from smpl.smpl_webuser.serialization import load_model

sys.path.append(os.path.abspath('../lib_py'))
from kinematics_lib_br import KinematicsLib

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')


class MeshDepthLib():

    def __init__(self, loss_type, batch_size, vertices):
        self.loss_type = loss_type

        self.dtype = torch.FloatTensor
        self.dtypeInt = torch.LongTensor

        if self.loss_type == 'anglesDC':
            self.bounds = torch.Tensor([
                [-0.5933865286111969, 0.5933865286111969],
                [-2*np.pi, 2*np.pi],
                [-1.215762200416361, 1.215762200416361],
                [-1.5793940868065197, 0.3097956806],
                [-0.5881754611, 0.5689768556],
                [-0.5323249722, 0.6736965222],
                [-1.5793940868065197, 0.3097956806],
                [-0.5689768556, 0.5881754611],
                [-0.6736965222, 0.5323249722],
                [-np.pi / 3, np.pi / 3],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 36, np.pi / 36],
                [-0.02268926111, 2.441713561],
                [-0.01, 0.01],
                [-0.01, 0.01],    # knee
                [-0.02268926111, 2.441713561],
                [-0.01, 0.01], [-0.01, 0.01],
                [-np.pi / 3, np.pi / 3],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                # ankle, pi/36 or 5 deg
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                # ankle, pi/36 or 5 deg
                [-np.pi / 3, np.pi / 3],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 36, np.pi / 36],
                [-0.01, 0.01],
                [-0.01, 0.01],
                [-0.01, 0.01],    # foot
                [-0.01, 0.01],
                [-0.01, 0.01],
                [-0.01, 0.01],    # foot
                [-np.pi / 3, np.pi / 3],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 36, np.pi / 36],    # neck
                [-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],
                [-2.455676183 * 1 / 3, 0.7627082389 * 1 / 3],
                [-1.570795 * 1 / 3, 2.188641033 * 1 / 3],
                [-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],
                [-0.7627082389 * 1 / 3, 2.455676183 * 1 / 3],
                [-2.188641033 * 1 / 3, 1.570795 * 1 / 3],
                [-np.pi / 3, np.pi / 3],
                [-np.pi / 36, np.pi / 36],
                [-np.pi / 36, np.pi / 36],    # head
                [-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],
                [-2.455676183 * 2 / 3, 0.7627082389 * 2 / 3],
                [-1.570795 * 2 / 3, 2.188641033 * 2 / 3],
                [-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],
                [-0.7627082389 * 2 / 3, 2.455676183 * 2 / 3],
                [-2.188641033 * 2 / 3, 1.570795 * 2 / 3],
                [-0.01, 0.01],
                [-2.570867817, 0.04799651389],
                [-0.01, 0.01],    # elbow
                [-0.01, 0.01],
                [-0.04799651389, 2.570867817],
                [-0.01, 0.01],    # elbow
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                # wrist, pi/36 or 5 deg
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                [-np.pi / 6, np.pi / 6],
                # wrist, pi/36 or 5 deg
                [-0.01, 0.01], [-0.01, 0.01],
                [-0.01, 0.01],    # hand
                [-0.01, 0.01],
                [-0.01, 0.01],
                [-0.01, 0.01]
            ]) * 1.2

        elif self.loss_type == 'anglesEU':
            self.bounds = torch.Tensor([
                [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 3, np.pi / 3],
                # [np.deg2rad(-90.0), np.deg2rad(17.8)], [np.deg2rad(-33.7), np.deg2rad(32.6)], [np.deg2rad(-30.5), np.deg2rad(38.6)],
                [-2.753284558994594, -0.2389229307048895],
                [-1.0047479181618846, 0.8034397361593714],
                [-0.8034397361593714, 1.0678805158941416],
                # [np.deg2rad(-90.0), np.deg2rad(17.8)], [np.deg2rad(-32.6), np.deg2rad(33.7)], [np.deg2rad(-38.6), np.deg2rad(30.5)],
                [-2.753284558994594, -0.2389229307048895],
                [-0.8034397361593714, 1.0047479181618846],
                [-1.0678805158941416, 0.8034397361593714],
                [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                # [np.deg2rad(-1.3), np.deg2rad(139.9)], [-0.01, 0.01], [-0.01, 0.01]
                [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],    # knee
                # [np.deg2rad(-1.3), np.deg2rad(139.9)], [-0.01, 0.01], [-0.01, 0.01]
                [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],
                [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                # ankle, pi/36 or 5 deg
                [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                # ankle, pi/36 or 5 deg
                [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],    # foot
                [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],    # foot
                [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # neck
                # [np.deg2rad(-88.9) * 1 / 3, np.deg2rad(81.4) * 1 / 3],
                [-1.5704982490935508 * 1 / 3, 1.6731615204412293 * 1 / 3],
                # [np.deg2rad(-140.7) * 1 / 3, np.deg2rad(43.7) * 1 / 3],
                [-1.5359250989762832 * 1 / 3, 0.4892616775215104 * 1 / 3],
                # [np.deg2rad(-135.0) * 1 / 3, np.deg2rad(80.4) * 1 / 3],
                [-2.032907094968176 * 1 / 3, 1.927742086422412 * 1 / 3],
                # [np.deg2rad(-88.9) * 1 / 3, np.deg2rad(81.4) * 1 / 3],
                [-1.5704982490935508 * 1 / 3, 1.6731615204412293 * 1 / 3],
                # [np.deg2rad(-43.7) * 1 / 3, np.deg2rad(140.7) * 1 / 3],
                [-0.4892616775215104 * 1 / 3, 1.5359250989762832 * 1 / 3],
                # [np.deg2rad(-80.4) * 1 / 3, np.deg2rad(135.0) * 1 / 3],
                [-1.927742086422412 * 1 / 3, 2.032907094968176 * 1 / 3],
                [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # head
                # [np.deg2rad(-88.9) * 2 / 3, np.deg2rad(81.4) * 2 / 3],
                [-1.5704982490935508 * 2 / 3, 1.6731615204412293 * 2 / 3],
                # [np.deg2rad(-140.7) * 2 / 3, np.deg2rad(43.7) * 2 / 3],
                [-1.5359250989762832 * 2 / 3, 0.4892616775215104 * 2 / 3],
                # [np.deg2rad(-135.0) * 2 / 3, np.deg2rad(80.4) * 2 / 3],
                [-2.032907094968176 * 2 / 3, 1.927742086422412 * 2 / 3],
                # [np.deg2rad(-88.9) * 2 / 3, np.deg2rad(81.4) * 2 / 3],
                [-1.5704982490935508 * 2 / 3, 1.6731615204412293 * 2 / 3],
                # [np.deg2rad(-43.7) * 2 / 3, np.deg2rad(140.7) * 2 / 3],
                [-0.4892616775215104 * 2 / 3, 1.5359250989762832 * 2 / 3],
                # [np.deg2rad(-80.4) * 2 / 3, np.deg2rad(135.0) * 2 / 3],
                [-1.927742086422412 * 2 / 3, 2.032907094968176 * 2 / 3],
                # [-0.01, 0.01], [np.deg2rad(-147.3), np.deg2rad(2.8)], [-0.01, 0.01],
                [-0.01, 0.01], [-2.463868908637374, 0.0], [-0.01, 0.01],    # elbow
                # [-0.01, 0.01], [np.deg2rad(-2.8), np.deg2rad(147.3)], [-0.01, 0.01],
                [-0.01, 0.01], [0.0, 2.463868908637374], [-0.01, 0.01],     # elbow
                [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                # wrist, pi/36 or 5 deg
                [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                # wrist, pi/36 or 5 deg
                [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],    # hand
                [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]
            ])

        SMPL_path_female = '../smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        SMPL_model_female = load_model(SMPL_path_female)

        self.SMPL_v_template_f = torch.Tensor(np.array(SMPL_model_female.v_template))
        self.SMPL_shapedirs_f = torch.Tensor(np.array(SMPL_model_female.shapedirs)).permute(0, 2, 1)
        self.SMPL_J_regressor_f = np.zeros((SMPL_model_female.J_regressor.shape)) + SMPL_model_female.J_regressor
        self.SMPL_J_regressor_f = torch.Tensor(np.array(self.SMPL_J_regressor_f).astype(float)).permute(1, 0)
        self.SMPL_posedirs_f = torch.Tensor(np.array(SMPL_model_female.posedirs))
        self.SMPL_weights_f = torch.Tensor(np.array(SMPL_model_female.weights))

        SMPL_path_male = '../smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        SMPL_model_male = load_model(SMPL_path_male)
        self.SMPL_v_template_m = torch.Tensor(np.array(SMPL_model_male.v_template))
        self.SMPL_shapedirs_m = torch.Tensor(np.array(SMPL_model_male.shapedirs)).permute(0, 2, 1)
        self.SMPL_J_regressor_m = np.zeros((SMPL_model_male.J_regressor.shape)) + SMPL_model_male.J_regressor
        self.SMPL_J_regressor_m = torch.Tensor(np.array(self.SMPL_J_regressor_m).astype(float)).permute(1, 0)
        self.SMPL_posedirs_m = torch.Tensor(np.array(SMPL_model_male.posedirs))
        self.SMPL_weights_m = torch.Tensor(np.array(SMPL_model_male.weights))


        if vertices == "all":   # mod 2

            self.parents = np.array(
                [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(
                np.int32)

            if batch_size == 128:
                batch_sub_divider = 8
            elif batch_size == 64:
                batch_sub_divider = 4
            elif batch_size == 32:
                batch_sub_divider = 2
            else:
                batch_sub_divider = 1

            self.batch_size = int(batch_size / batch_sub_divider)
            self.SMPL_shapedirs_f = self.SMPL_shapedirs_f.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.SMPL_shapedirs_m = self.SMPL_shapedirs_m.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.shapedirs = torch.cat((self.SMPL_shapedirs_f, self.SMPL_shapedirs_m), 0)  # this is 2 x N x B x R x D
            self.SMPL_B = self.shapedirs.size()[2]  # this is 10
            self.SMPL_R = self.shapedirs.size()[3]  # this is 6890, or num of verts
            self.SMPL_D = self.shapedirs.size()[4]  # this is 3, or num dimensions
            self.R_used = 6890
            self.shapedirs = self.shapedirs.permute(1, 0, 2, 3, 4).view(self.batch_size, 2, self.SMPL_B * self.SMPL_R * self.SMPL_D)


            self.SMPL_v_template_f = self.SMPL_v_template_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_v_template_m = self.SMPL_v_template_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.v_template = torch.cat((self.SMPL_v_template_f, self.SMPL_v_template_m), 0)  # this is 2 x N x R x D
            self.v_template = self.v_template.permute(1, 0, 2, 3).view(self.batch_size, 2, self.SMPL_R * self.SMPL_D)


            self.J_regressor = torch.cat((self.SMPL_J_regressor_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0),
                                            self.SMPL_J_regressor_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)),
                                            0)  # this is 2 x N x R x 24
            self.J_regressor = self.J_regressor.permute(1, 0, 2, 3).view(self.batch_size, 2, self.SMPL_R * 24)


            self.posedirs = torch.cat((self.SMPL_posedirs_f.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).unsqueeze(0),
                                        self.SMPL_posedirs_m.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).unsqueeze(0)), 0)
            # self.posedirs = self.posedirs.permute(1, 0, 2, 3, 4).view(self.N, 2, self.SMPL_R*self.SMPL_D*207)
            self.posedirs = self.posedirs.permute(1, 0, 2, 3, 4).view(self.batch_size, 2, self.R_used * self.SMPL_D * 207)

            self.SMPL_weights_repeat_f = self.SMPL_weights_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_weights_repeat_m = self.SMPL_weights_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_weights_repeat = torch.cat((self.SMPL_weights_repeat_f, self.SMPL_weights_repeat_m), 0)
            # self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.SMPL_R * 24)
            self.SMPL_weights_repeat = self.SMPL_weights_repeat.permute(1, 0, 2, 3).view(self.batch_size, 2, self.R_used * 24)

            if self.loss_type == 'anglesEU':
                self.zeros_cartesian = torch.zeros([batch_size, 24])
                self.ones_cartesian = torch.ones([batch_size, 24])

            self.filler_taxels = []
            for i in range(28):
                for j in range(65):
                    self.filler_taxels.append([i - 1, j - 1, 20000])
            self.filler_taxels = torch.Tensor(self.filler_taxels).type(self.dtypeInt).unsqueeze(0).repeat(
                batch_size, 1, 1)
            self.mesh_patching_array = torch.zeros((batch_size, 66, 29, 4)).type(self.dtype)

        else:                   # mod 1
            self.parents = np.array(
                [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(
                np.int32)

            self.batch_size = batch_size
            self.SMPL_shapedirs_repeat_f = self.SMPL_shapedirs_f.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).permute(0, 2, 1,
                                                                                                    3).unsqueeze(0)
            self.SMPL_shapedirs_repeat_m = self.SMPL_shapedirs_m.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).permute(0, 2, 1,
                                                                                                    3).unsqueeze(0)
            self.SMPL_shapedirs_repeat = torch.cat((self.SMPL_shapedirs_repeat_f, self.SMPL_shapedirs_repeat_m),
                                                0)  # this is 2 x N x B x R x D
            self.SMPL_B = self.SMPL_shapedirs_repeat.size()[2]  # this is 10
            self.SMPL_R = self.SMPL_shapedirs_repeat.size()[3]  # this is 6890, or num of verts
            self.SMPL_D = self.SMPL_shapedirs_repeat.size()[4]  # this is 3, or num dimensions
            self.SMPL_shapedirs_repeat = self.SMPL_shapedirs_repeat.permute(1, 0, 2, 3, 4).view(self.batch_size, 2,
                                                                                        self.SMPL_B * self.SMPL_R * self.SMPL_D)

            self.SMPL_v_template_repeat_f = self.SMPL_v_template_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_v_template_repeat_m = self.SMPL_v_template_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_v_template_repeat = torch.cat((self.SMPL_v_template_repeat_f, self.SMPL_v_template_repeat_m),
                                                0)  # this is 2 x N x R x D
            self.SMPL_v_template_repeat = self.SMPL_v_template_repeat.permute(1, 0, 2, 3).view(self.batch_size, 2, self.SMPL_R * self.SMPL_D)

            self.SMPL_J_regressor_repeat_f = self.SMPL_J_regressor_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_J_regressor_repeat_m = self.SMPL_J_regressor_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_J_regressor_repeat = torch.cat((self.SMPL_J_regressor_repeat_f, self.SMPL_J_regressor_repeat_m),
                                                0)  # this is 2 x N x R x 24
            self.SMPL_J_regressor_repeat = self.SMPL_J_regressor_repeat.permute(1, 0, 2, 3).view(self.batch_size, 2, self.SMPL_R * 24)

            self.SMPL_posedirs_repeat_f = self.SMPL_posedirs_f.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).unsqueeze(0)
            self.SMPL_posedirs_repeat_m = self.SMPL_posedirs_m.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).unsqueeze(0)
            self.SMPL_posedirs_repeat = torch.cat((self.SMPL_posedirs_repeat_f, self.SMPL_posedirs_repeat_m), 0)
            self.SMPL_posedirs_repeat = self.SMPL_posedirs_repeat.permute(1, 0, 2, 3, 4).view(self.batch_size, 2, self.SMPL_R*self.SMPL_D*207)
            # self.SMPL_posedirs_repeat = self.SMPL_posedirs_repeat.permute(1, 0, 2, 3, 4).view(self.batch_size, 2, 10 * self.SMPL_D * 207)  # self.num_posedirs

            self.SMPL_weights_repeat_f = self.SMPL_weights_f.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_weights_repeat_m = self.SMPL_weights_m.unsqueeze(0).repeat(self.batch_size, 1, 1).unsqueeze(0)
            self.SMPL_weights_repeat = torch.cat((self.SMPL_weights_repeat_f, self.SMPL_weights_repeat_m), 0)
            # self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.SMPL_R * 24)
            self.SMPL_weights_repeat = self.SMPL_weights_repeat.permute(1, 0, 2, 3).view(self.batch_size, 2, self.SMPL_R * 24)
            if self.loss_type == 'anglesEU':
                self.zeros_cartesian = torch.zeros([self.batch_size, 24])
                self.ones_cartesian = torch.ones([self.batch_size, 24])


    #human mesh recovery - kinematic embedding
    def HMR(self, gender_switch, betas_est, Rs_est, root_shift_est, start_incr, end_incr, GPU):

        if GPU == False:
            self.dtype = torch.FloatTensor
            self.dtypeInt = torch.LongTensor

        sub_batch_size = end_incr - start_incr

        shapedirs = torch.bmm(gender_switch[start_incr:end_incr, :, :],
                              self.shapedirs[0:sub_batch_size, :, :]) \
            .view(sub_batch_size, self.SMPL_B, self.SMPL_R * self.SMPL_D)

        betas_shapedirs_mult = torch.bmm(betas_est[start_incr:end_incr, :].unsqueeze(1), shapedirs) \
            .squeeze(1) \
            .view(sub_batch_size, self.SMPL_R, self.SMPL_D)

        v_template = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.v_template[0:sub_batch_size, :, :]) \
            .view(sub_batch_size, self.SMPL_R, self.SMPL_D)

        v_shaped = betas_shapedirs_mult + v_template

        J_regressor = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.J_regressor[0:sub_batch_size, :, :]) \
            .view(sub_batch_size, self.SMPL_R, 24)

        Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor).squeeze(1)
        Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor).squeeze(1)
        Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor).squeeze(1)


        J_est = torch.stack([Jx, Jy, Jz],
                            dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
        # J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

        targets_est, A_est = KinematicsLib().batch_global_rigid_transformation(Rs_est[start_incr:end_incr, :],
                                                                               J_est, self.parents, GPU,
                                                                               rotate_base=False)


        targets_est = targets_est + root_shift_est[start_incr:end_incr, :].unsqueeze(1) - J_est[:, 0:1, :]

        pose_feature = (Rs_est[start_incr:end_incr, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)

        #print(torch.cuda.max_memory_allocated(), 'meshdepthlib4', gender_switch.size(), betas_est.size(), Rs_est.size(), root_shift_est.size(), start_incr, end_incr)

        posedirs = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.posedirs[0:sub_batch_size, :, :]) \
            .view(sub_batch_size, self.R_used * self.SMPL_D, 207) \
            .permute(0, 2, 1)

        #print(torch.cuda.max_memory_allocated(), 'meshdepthlib5', gender_switch.size(), betas_est.size(), Rs_est.size(), root_shift_est.size(), start_incr, end_incr)

        v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs).view(-1, self.R_used, self.SMPL_D)

        v_posed = v_posed.clone() + v_shaped

        weights_repeat = torch.bmm(gender_switch[start_incr:end_incr, :, :],
                                   self.SMPL_weights_repeat[0:sub_batch_size, :, :]) \
            .squeeze(1) \
            .view(sub_batch_size, self.R_used, 24)
        T = torch.bmm(weights_repeat, A_est.view(sub_batch_size, 24, 16)).view(sub_batch_size, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(sub_batch_size, v_posed.shape[1], 1).type(self.dtype)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0] + root_shift_est[start_incr:end_incr, :].unsqueeze(1) - J_est[:, 0:1, :]


        return verts, J_est, targets_est


    #PMR - Pressure Map Reconstruction#
    def PMR(self, verts, bed_angle_batch, get_mesh_bottom_dist = True):
        cbs = verts.size()[0] #current batch size
        bend_taxel_loc = 48

        #verts[:, :, 1] -= 10*0.0286

        bed_angle_batch = bed_angle_batch.mul(np.pi/180)
        bed_angle_batch_sin = torch.sin(bed_angle_batch)
        bed_angle_batch_cos = torch.cos(bed_angle_batch)


        #compute the depth and contact maps from the mesh
        verts_taxel = verts.clone()
        verts_rot_taxel = verts.clone()

        bend_loc = bend_taxel_loc*0.0286

        verts_rot_taxel[:, :, 1] = torch.mul(bed_angle_batch_sin, verts_taxel[:, :, 2].permute(1, 0)).permute(1, 0) - \
                                   torch.mul(bed_angle_batch_cos, (bend_loc - verts_taxel[:, :, 1]).permute(1, 0)).permute(1, 0) + bend_loc

        verts_rot_taxel[:, :, 2] = torch.mul(bed_angle_batch_cos, verts_taxel[:, :, 2].permute(1, 0)).permute(1, 0) + \
                                   torch.mul(bed_angle_batch_sin, (bend_loc - verts_taxel[:, :, 1]).permute(1, 0)).permute(1, 0)

        verts_taxel = torch.cat((verts_taxel, verts_taxel[:, 0:8000, :]*0+3.0), dim = 1)
        #print verts_taxel.size(),"SIZE POST"


        for i in range(cbs):
            body_verts = verts_taxel[i, verts_taxel[i, :, 1] < bend_loc]
            head_verts = verts_rot_taxel[i, verts_rot_taxel[i, :, 1] >= bend_loc]

            verts_taxel[i, 0:body_verts.size()[0], :] = body_verts
            #print verts_taxel[i, body_verts.size()[0]:(body_verts.size()[0]+head_verts.size()[0]), :].size()
            #print verts_taxel[i, :, :].size()
            verts_taxel[i, body_verts.size()[0]:(body_verts.size()[0]+head_verts.size()[0]), :] = head_verts
            verts_taxel[i, body_verts.size()[0]+head_verts.size()[0]:, :] *= 0


        #plt.plot(-verts_taxel.cpu().detach().numpy()[0, :, 1], verts_taxel.cpu().detach().numpy()[0, :, 2], 'k.')

        #plt.axis([-1.8, -0.2, -0.3, 1.0])
       # plt.show()

        verts_taxel /= 0.0286
        verts_taxel[:, :, 0] -= 10
        verts_taxel[:, :, 1] -= 10
        verts_taxel[:, :, 2] *= 1000
        verts_taxel[:, :, 0] *= 1.04

        verts_taxel_int = (verts_taxel).type(self.dtypeInt)

        #3print self.filler_taxels.shape, 'filler shape'
        if get_mesh_bottom_dist == False:
            print("GETTING THE TOP MESH DIST")
            verts_taxel_int[:, :, 2] *= -1

        verts_taxel_int = torch.cat((self.filler_taxels[0:cbs, :, :], verts_taxel_int), dim=1)

        vertice_sorting_method = (verts_taxel_int[:, :, 0:1] + 1) * 10000000 + \
                                 (verts_taxel_int[:, :, 1:2] + 1) * 100000 + \
                                 verts_taxel_int[:, :, 2:3]
        verts_taxel_int = torch.cat((vertice_sorting_method, verts_taxel_int), dim=2)
        for i in range(cbs):
            t1 = time.time()

            #for k in range(verts_taxel_int[i, :, :].shape[0]):
            #    print verts_taxel_int[i, k, :]

            #print torch.min(verts_taxel_int[i, :, 3]), torch.max(verts_taxel_int[i, :, 3]), verts_taxel_int[i, :, :].shape

            x = torch.unique(verts_taxel_int[i, :, :], sorted=True, return_inverse=False,
                             dim=0)  # this takes the most time


            t2 = time.time()


            x[1:, 0] = torch.abs((x[:-1, 1] - x[1:, 1]) + (x[:-1, 2] - x[1:, 2]))
            x[1:, 1:4] = x[1:, 1:4] * x[1:, 0:1]
            x = x[x[:, 0] != 0, :]
            x = x[:, 1:]
            x = x[x[:, 1] < 64, :]
            x = x[x[:, 1] >= 0, :]
            x = x[x[:, 0] < 27, :]
            x = x[x[:, 0] >= 0, :]


            #print torch.min(x[:, 2]), torch.max(x[:, 2]), x.shape
            mesh_matrix = x[:, 2].view(27, 64)

            if i == 0:
                # print mesh_matrix[0:15, 32:]
                mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
                # print mesh_matrix[0, 1:32, 0:15]
                mesh_matrix_batch = mesh_matrix.clone()
            else:
                mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
                mesh_matrix_batch = torch.cat((mesh_matrix_batch, mesh_matrix), dim=0)

            t3 = time.time()
        # print i, t3 - t2, t2 - t1

        mesh_matrix_batch[mesh_matrix_batch == 20000] = 0

        mesh_matrix_batch = mesh_matrix_batch.type(self.dtype)
        mesh_matrix_batch *= 0.0286  # shouldn't need this. leave as int.

        self.mesh_patching_array *= 0
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 0] = mesh_matrix_batch.clone()
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 0][self.mesh_patching_array[0:cbs, 1:65, 1:28, 0] > 0] = 0
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 0] = self.mesh_patching_array[0:cbs, 0:64, 0:27, 0] + \
                                                         self.mesh_patching_array[0:cbs, 1:65, 0:27, 0] + \
                                                         self.mesh_patching_array[0:cbs, 2:66, 0:27, 0] + \
                                                         self.mesh_patching_array[0:cbs, 0:64, 1:28, 0] + \
                                                         self.mesh_patching_array[0:cbs, 2:66, 1:28, 0] + \
                                                         self.mesh_patching_array[0:cbs, 0:64, 2:29, 0] + \
                                                         self.mesh_patching_array[0:cbs, 1:65, 2:29, 0] + \
                                                         self.mesh_patching_array[0:cbs, 2:66, 2:29, 0]
        self.mesh_patching_array[0:cbs, :, :, 0] /= 8
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 1] = mesh_matrix_batch.clone()
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 1][self.mesh_patching_array[0:cbs, 1:65, 1:28, 1] < 0] = 0
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 1][self.mesh_patching_array[0:cbs, 1:65, 1:28, 1] >= 0] = 1
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 2] = self.mesh_patching_array[0:cbs, 1:65, 1:28,
                                                         0] * self.mesh_patching_array[0:cbs, 1:65, 1:28, 1]
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 3] = self.mesh_patching_array[0:cbs, 1:65, 1:28, 2].clone()
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 3][self.mesh_patching_array[0:cbs, 1:65, 1:28, 3] != 0] = 1.
        self.mesh_patching_array[0:cbs, 1:65, 1:28, 3] = 1 - self.mesh_patching_array[0:cbs, 1:65, 1:28, 3]
        mesh_matrix_batch = mesh_matrix_batch * self.mesh_patching_array[0:cbs, 1:65, 1:28, 3]
        mesh_matrix_batch += self.mesh_patching_array[0:cbs, 1:65, 1:28, 2]
        mesh_matrix_batch = mesh_matrix_batch.type(self.dtypeInt)


        contact_matrix_batch = mesh_matrix_batch.clone()
        contact_matrix_batch[contact_matrix_batch >= 0] = 0
        contact_matrix_batch[contact_matrix_batch < 0] = 1

        #print mesh_matrix_batch

        #print torch.min(mesh_matrix_batch[0, :, :]), torch.max(mesh_matrix_batch[0, :, :]), "A"

        if get_mesh_bottom_dist == False:
            mesh_matrix_batch *= -1

        #print torch.min(mesh_matrix_batch[0, :, :]), torch.max(mesh_matrix_batch[0, :, :])

        return mesh_matrix_batch, contact_matrix_batch


# # Call the class
# mod = 2
# vertices_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
# if mod == 1:
#     mesh_depth_lib = MeshDepthLib(loss_type='anglesDC', batch_size=1, vertices=vertices_list)
# else:
#     mesh_depth_lib = MeshDepthLib(loss_type='anglesDC', batch_size=1, vertices='all')