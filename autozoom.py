#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
# import flask
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


dataset = 'TAMULF'
arguments_strIn = '/media/data/prasan/datasets/LF_datasets/'
arguments_strOut = 'TAMULF.mp4'
filenames_file = 'aryan_test_inputs/{}/test_files.txt'.format(dataset)
height = 256  # 
width = 192
doTime = True
color_corr = True

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################


if __name__ == '__main__':
	with open(filenames_file, 'r') as f:
		filenames = f.readlines()
	idxs = range(len(filenames))

	save_path = 'test_results'
	os.makedirs(save_path, exist_ok=True)
	f = open(os.path.join(save_path, f'{height}x{width}_{dataset}_results.txt'), 'w')
	psnr_avg = RunningAverage()
	ssim_avg = RunningAverage()
	all_times = []
	with tqdm(enumerate(idxs), total=len(idxs), desc='Testing - {}'.format(dataset)) as vepoch:
		for i, idx in vepoch:
			file_path = filenames[idx][:-1]
			# print(arguments_strIn, file_path)
			lf = np.load(os.path.join(arguments_strIn, file_path))
			# print(lf.shape)
			
			if color_corr:
				lf = lf / 255.
				mean = lf.mean()
				fact = np.log(0.4) / np.log(mean)
				if fact<1:
					lf = lf ** fact
				lf = np.uint8(255 * lf)

			if lf.shape[4] == 3:
				lf = lf.transpose([0, 1, 4, 2, 3])
			else:
				lf = lf.transpose([0, 1, 2, 3, 4])
			#print(lf.shape)
			X, Y, C, H, W = lf.shape
			lf = lf.reshape(X*Y, C, H, W)
			npyLightField = np.zeros([X*Y, height, width, C])
			
			for j in range(X*Y):
				img = lf[j]
				img = np.transpose(img, [1, 2, 0])
				img = cv2.resize(src=img, dsize=(width, height), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)
				npyLightField[j, ...] = img
			
			npyImage = npyLightField[X*Y//2, ...]

			intWidth = npyImage.shape[1]
			intHeight = npyImage.shape[0]
	
			fltRatio = float(intWidth) / float(intHeight)

			# intWidth = min(int(1024 * fltRatio), 1024)
			# intHeight = min(int(1024 / fltRatio), 1024)

			npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

			process_load(npyImage, {})
			# print('1. Loaded image')
			starttime = time.time()

			objFrom = {
				'fltCenterU': intWidth / 2.0,
				'fltCenterV': intHeight / 2.0,
				'intCropWidth': int(math.floor(0.97 * intWidth)),
				'intCropHeight': int(math.floor(0.97 * intHeight))
			}
			# print('1.5 objFrom done')

			objTo = process_autozoom({
				'fltShift': 100.0,
				'fltZoom': 1.25,
				'objFrom': objFrom
			})
			# print('2. Autozoom done')

			npyResult = process_kenburns({
				'fltSteps': numpy.linspace(0.0, 1, 49).tolist(),
				'objFrom': objFrom,
				'objTo': objTo,
				'boolInpaint': True
			})
			total_time = time.time() - starttime
			all_times.append(total_time)

			# print('3. 3D Ken Burns done')
			# print(np.array(npyResult).shape)
			
			# Use for saving video
			# moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:-1] ], fps=25).write_videofile(arguments_strOut)
			# moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyLightField], fps=25).write_videofile("orig_"+arguments_strOut)

			npyResult = np.array(npyResult)

			# npyReconsLightField = np.concatenate(npyResult, axis=0)
			# print(npyReconsLightField.shape, npyLightField.shape) # will fail here

			psnr = calculate_psnr(npyLightField, npyResult)
			ssim = calculate_ssim(npyLightField, npyResult)

			psnr_avg.append(psnr)
			ssim_avg.append(ssim)

			# lf_paths, img_paths = get_paths(save_path, i, 1)
			# imageio.imwrite(img_paths[0], np.uint8(npyImage))

			# save_video_from_lf(npyReconsLightField, lf_paths[0])

			vepoch.set_postfix(PSNR="{:0.4f}({:0.4f})".format(psnr_avg.get_value(), psnr),
								SSIM="{:0.4f}({:0.4f})".format(ssim_avg.get_value(), ssim))

			string = 'Sample {0:2d} => PSNR: {1:.4f}, SSIM: {2:.4f}, Time: {3:.4f}\n'.format(i, psnr, ssim, total_time)
			f.write(string)
			#break

	avg_psnr = psnr_avg.get_value()
	avg_ssim = ssim_avg.get_value()
	avg_time = np.mean(all_times)
	string = 'Average PSNR: {0:.4f}\nAverage SSIM: {1:.4f} Average Time: {2:.4f}\n'.format(avg_psnr, avg_ssim, avg_time)
	f.write(string)
	f.close()

	#moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=5).write_videofile(arguments_strOut)
# end