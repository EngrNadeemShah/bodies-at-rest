import sys
from matplotlib.pylab import *

import sys
sys.path.insert(0, '../lib_py')

MAT_WIDTH = 0.762   # metres
MAT_HEIGHT = 1.854	# metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84	# 73 # taxels
NUMOFTAXELS_Y = 47	# 30
INTER_SENSOR_DISTANCE = 0.0286	# metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


import pickle as pickle
def load_pickle(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f, encoding='latin1')

class UnpackBatchLib():
	def unpack_batch(self, inputs, labels, is_training, model, config):

		# # labels (y)
		# # batch[1]  |   0:72      - markers_xyz_m
		# # batch[2]  |   72:82     - body_shape
		# # batch[3]  |   82:154    - joint_angles
		# # batch[4]  |   154:157   - root_xyz_shift
		# # batch[5]  |   157:159   - g1, g2 (gender switch)
		# # batch[6]  |   159       - s1 (synth vs real)
		# # batch[7]  |   160       - body_mass
		# # batch[8]  |   161       - body_height

		# # when mod 2 (adjust_ang_from_est) is true, we have the following additional labels
		# # batch[9]  |   162:172   - betas_est
		# # batch[10] |   172:244   - angles_est
		# # batch[11] |   244:247   - root_xyz_est
		# # batch[12] |   247:253   - root_atan2_est

		# # inputs (x)
		# # batch[0]  |   0:3 or 0:6
		#     # mod 1: 0: PM Contact, 1: PM, 2: PM Sobel
		#     # mod 2: 0: PM Contact, 1: DM Est +, 2: DM Est -, 3: CM Est, 4: PM, 5: PM Sobel
		# # batch[13] |   depth_map
		# # batch[14] |   contact_map

		if config['adjust_ang_from_est']:
			OUTPUT_EST_DICT = {
				'betas_est': labels[:, 162:172],
				'angles_est': labels[:, 172:244],
				'root_xyz_est': labels[:, 244:247],
				'root_atan2_est': labels[:, 247:253],
			}

			INPUT_DICT = {
				'depth_map': inputs[:, -1, :, :],
				'contact_map': inputs[:, -2, :, :],
			}

		else:
			OUTPUT_EST_DICT = None
			INPUT_DICT = {
				'depth_map': None,
				'contact_map': None,
			}

		inputs = inputs[:, :-2, :, :] if config['depth_map_labels'] else inputs

		predicted_labels, OUTPUT_DICT = model.forward_kinematic_angles(inputs		= inputs,
															 label_markers_xyz		= labels[:, :72],
															 label_body_shape		= labels[:, 72:82],
															 label_joint_angles		= labels[:, 82:154],
															 label_root_xyz			= labels[:, 154:157],
															 label_gender_switch	= labels[:, 157:159],
															 label_synth_real_switch= labels[:, 159],
															 config					= config,
															 OUTPUT_EST_DICT		= OUTPUT_EST_DICT,
															 is_training			= is_training)	# scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.


		INPUT_DICT['x_images'] = inputs
		INPUT_DICT['y_true_markers_xyz'] = labels[:, :72]

		return predicted_labels, INPUT_DICT, OUTPUT_DICT