#!/usr/bin/env python

import torch
import torchvision

import argparse
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import imageio
import math
import moviepy
import moviepy.editor
import numpy
import numpy as np
import os
from PIL import Image
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile
from tqdm import tqdm

from metrics import SSIM
from utils import RunningAverage
from tensor_ops import get_paths, save_img, save_video_from_lf
from loss_temp import *

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################


def calculate_psnr(img1, img2):
	img1 = img1#.cpu()
	img2 = img2#.cpu()
	mse = np.mean((img1 - img2)**2)
	if mse == 0:
		return float('inf')
	return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    with torch.no_grad():
        ssim = SSIM()
        V, C, H, W = img1.shape
        img1 = torch.tensor(img1)/255.
        img2 = torch.tensor(img2)/255.
        ssim = ssim(img1, img2).numpy()
        return ssim


##########################################################


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


# Arguments
parser = argparse.ArgumentParser(description='Testing script. Default values of all arguments are recommended for reproducibility', 
                                    fromfile_prefix_chars='@', conflict_handler='resolve')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')

######################################## Dataset parameters #######################################
parser.add_argument('-d', '--dataset', default='Kalantari', type=str, help='Dataset to train on')

############################################# I/0 parameters ######################################
#parser.add_argument('-th', '--train_height', type=int, help='input height', default=180)
#parser.add_argument('-tw', '--train_width', type=int, help='input width', default=270)
#parser.add_argument('-vh', '--val_height', type=int, help='input height', default=352)
#parser.add_argument('-vw', '--val_width', type=int, help='input width', default=528)
parser.add_argument('-md', '--max_displacement', default=1.2, type=float)
parser.add_argument('-zp', '--zero_plane', default=0.3, type=float)
parser.add_argument('-cc', '--color_corr', default=True, action='store_true')

####################################### RAFT parameters ###########################################
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

##################################### Learning parameters #########################################
parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')

##################################### Tensor Display parameters #########################################
parser.add_argument('--angular', default= 7, type=int, help='number of angular views to output')

args = parser.parse_args()


dataset = args.dataset
arguments_strIn = '/media/data/prasan/datasets/LF_video_datasets/'
arguments_depth = '/media/data/prasan/datasets/LF_video_datasets/DPT-depth'
filenames_file = 'temp_inputs/{}/test_files.txt'.format(dataset)
color_corr = args.color_corr

device = torch.device('cuda:0')
temporal_loss = TemporalConsistency(args, device)

#for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
#	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################


if __name__ == '__main__':
    with open(filenames_file, 'r') as f:
        filenames = f.readlines()
    idxs = range(len(filenames))

    f = open(f'{dataset}_results.txt', 'w')
    temp_loss_avg = RunningAverage()

    with tqdm(enumerate(idxs), total=len(idxs), desc='Testing - {}'.format(dataset)) as vepoch:
        for i, idx in vepoch:
            file_paths = filenames[idx][:-1]
            prev_path, curr_path = file_paths.split(',')
            prev_lf = np.load(os.path.join(arguments_strIn, prev_path))
            lf = np.load(os.path.join(arguments_strIn, curr_path))

            if color_corr:
                lf = lf / 255.
                mean = lf.mean()
                fact = np.log(0.4) / np.log(mean)
                if fact<1:
                    lf = lf ** fact
                lf = np.uint8(255 * lf)

                prev_lf = prev_lf / 255.
                if fact<1:
                    prev_lf = prev_lf ** fact
                prev_lf = np.uint8(255 * prev_lf)

            if lf.shape[4] == 3:
                lf = lf.transpose([0, 1, 4, 2, 3])
                prev_lf = prev_lf.transpose([0, 1, 4, 2, 3])
            else:
                lf = lf.transpose([1, 0, 2, 3, 4])
                prev_lf = prev_lf.transpose([1, 0, 2, 3, 4])
            #print(lf.shape)
            X, Y, C, H, W = lf.shape
            lf = lf.reshape(X*Y, C, H, W)
            prev_lf = prev_lf.reshape(X*Y, C, H, W)
            npyLightField = np.zeros([X*Y, 352, 528, C])
            npyPrevLightField = np.zeros([X*Y, 352, 528, C])
			
            for j in range(X*Y):
                img = lf[j]
                img = np.transpose(img, [1, 2, 0])
                img = cv2.resize(src=img, dsize=(528, 352), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)
                npyLightField[j, ...] = img

                img = prev_lf[j]
                img = np.transpose(img, [1, 2, 0])
                img = cv2.resize(src=img, dsize=(528, 352), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)
                npyPrevLightField[j, ...] = img
			
            npyImage = npyLightField[X*Y//2, ...]

            intWidth = npyImage.shape[1]
            intHeight = npyImage.shape[0]

            fltRatio = float(intWidth) / float(intHeight)

            intWidth = 1024 # min(int(1024 * fltRatio), 1024)
            intHeight = 768 # min(int(1024 / fltRatio), 1024)

            npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

            process_load(npyImage, {})

            objFrom = {
                'fltCenterU': intWidth / 2.0,
                'fltCenterV': intHeight / 2.0,
                'intCropWidth': int(math.floor(0.97 * intWidth)),
                'intCropHeight': int(math.floor(0.97 * intHeight))
            }

            objTo = process_autozoom({
                'fltShift': 50.0,
                'fltZoom': 1.00,
                'objFrom': objFrom
            })

            npyResult = process_kenburns({
                'fltSteps': np.linspace(0, 1, args.angular).tolist(),
                'objFrom': objFrom,
                'objTo': objTo,
                'boolInpaint': True
            })
            npyReconsLightField = np.concatenate(npyResult, axis=0)

            ReconsLightField = torch.tensor(npyReconsLightField, dtype=torch.float).unsqueeze(0).to(device)
            ReconsLightField = ReconsLightField.permute([0, 1, 4, 2, 3])/255.
            LightField = torch.tensor(npyLightField, dtype=torch.float).unsqueeze(0).to(device)
            LightField = LightField.permute([0, 1, 4, 2, 3])/255.
            PrevLightField = torch.tensor(npyPrevLightField, dtype=torch.float).unsqueeze(0).to(device)
            PrevLightField = PrevLightField.permute([0, 1, 4, 2, 3])/255.
            
            temp_loss = temporal_loss(LightField, PrevLightField, ReconsLightField)
            temp_loss_avg.append(temp_loss)

            vepoch.set_postfix(temp_loss="{:0.6f}({:0.6f})".format(temp_loss_avg.get_value(), temp_loss))

            string = 'Sample {0:2d} => Temp loss: {1:.6f}\n'.format(i, temp_loss)
            f.write(string)

            #break

    avg_temp_loss = temp_loss_avg.get_value()
    string = 'Average Temp loss: {0:.6f}\n'.format(avg_temp_loss)
    f.write(string)
    f.close()

	#moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=5).write_videofile(arguments_strOut)
# end