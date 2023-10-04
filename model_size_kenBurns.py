#!/usr/bin/env python

import os
import sys
import cv2

import torch
import torchvision
from flopth import flopth

class Basic(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super().__init__()

		if strType == 'relu-conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.netShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tenInput):
		if self.netShortcut is None:
			return self.netMain(tenInput) + tenInput

		elif self.netShortcut is not None:
			return self.netMain(tenInput) + self.netShortcut(tenInput)

		# end
	# end
# end

class Downsample(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Upsample(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Semantics(torch.nn.Module):
	def __init__(self):
		super().__init__()

		netVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval()

		self.netVgg = torch.nn.Sequential(
			netVgg[0:3],
			netVgg[3:6],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			netVgg[7:10],
			netVgg[10:13],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			netVgg[14:17],
			netVgg[17:20],
			netVgg[20:23],
			netVgg[23:26],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			netVgg[27:30],
			netVgg[30:33],
			netVgg[33:36],
			netVgg[36:39],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
		)
	# end

	def forward(self, tenInput):
		tenPreprocessed = tenInput[:, [ 2, 1, 0 ], :, :]

		tenPreprocessed[:, 0, :, :] = (tenPreprocessed[:, 0, :, :] - 0.485) / 0.229
		tenPreprocessed[:, 1, :, :] = (tenPreprocessed[:, 1, :, :] - 0.456) / 0.224
		tenPreprocessed[:, 2, :, :] = (tenPreprocessed[:, 2, :, :] - 0.406) / 0.225

		return self.netVgg(tenPreprocessed)
	# end
# end

class Disparity(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.netImage = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
		self.netSemantics = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

		for intRow, intFeatures in [ (0, 32), (1, 48), (2, 64), (3, 512), (4, 512), (5, 512) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 48, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 64, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol), Downsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol), Downsample([ 512, 512, 512 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 512, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 64, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 48, 32, 32 ]))
		# end

		self.netDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tenImage, tenSemantics):
		tenColumn = [ None, None, None, None, None, None ]

		tenColumn[0] = self.netImage(tenImage)
		tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
		tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
		tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2]) + self.netSemantics(tenSemantics)
		tenColumn[4] = self._modules['3x0 - 4x0'](tenColumn[3])
		tenColumn[5] = self._modules['4x0 - 5x0'](tenColumn[4])

		intColumn = 1
		for intRow in range(len(tenColumn)):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != 0:
				tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		return torch.nn.functional.threshold(input=self.netDisparity(tenColumn[0]), threshold=0.0, value=0.0)
	# end
# end


# disp_adjustment:
netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# disp estimation
netSemantics = Semantics()
netDisparity = Disparity()
netDisparity.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/kenburns/network-disparity.pytorch', file_name='kenburns-disparity').items() })


class Basic2(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super().__init__()

		if strType == 'relu-conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.netShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tenInput):
		if self.netShortcut is None:
			return self.netMain(tenInput) + tenInput

		elif self.netShortcut is not None:
			return self.netMain(tenInput) + self.netShortcut(tenInput)

		# end
	# end
# end

class Downsample2(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Upsample2(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Refine(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.netImageOne = Basic2('conv-relu-conv', [ 3, 24, 24 ])
		self.netImageTwo = Downsample2([ 24, 48, 48 ])
		self.netImageThr = Downsample2([ 48, 96, 96 ])

		self.netDisparityOne = Basic2('conv-relu-conv', [ 1, 96, 96 ])
		self.netDisparityTwo = Upsample2([ 192, 96, 96 ])
		self.netDisparityThr = Upsample2([ 144, 48, 48 ])
		self.netDisparityFou = Basic2('conv-relu-conv', [ 72, 24, 24 ])

		self.netRefine = Basic2('conv-relu-conv', [ 24, 24, 1 ])
	# end

	def forward(self, tenImage, tenDisparity):
		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenImageOne = self.netImageOne(tenImage)
		tenImageTwo = self.netImageTwo(tenImageOne)
		tenImageThr = self.netImageThr(tenImageTwo)

		tenUpsample = self.netDisparityOne(tenDisparity)
		if tenUpsample.shape != tenImageThr.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageThr.shape[2], tenImageThr.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityTwo(torch.cat([ tenImageThr, tenUpsample ], 1)); tenImageThr = None
		if tenUpsample.shape != tenImageTwo.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageTwo.shape[2], tenImageTwo.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityThr(torch.cat([ tenImageTwo, tenUpsample ], 1)); tenImageTwo = None
		if tenUpsample.shape != tenImageOne.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageOne.shape[2], tenImageOne.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityFou(torch.cat([ tenImageOne, tenUpsample ], 1)); tenImageOne = None

		tenRefine = self.netRefine(tenUpsample)
		tenRefine *= tenStd[1] + 0.0000001
		tenRefine += tenMean[1]

		return torch.nn.functional.threshold(input=tenRefine, threshold=0.0, value=0.0)
	# end
# end

# refinement
netRefine = Refine()
netRefine.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/kenburns/network-refinement.pytorch', file_name='kenburns-refinement').items() })


class Basic3(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super().__init__()

		if strType == 'relu-conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.netShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tenInput):
		if self.netShortcut is None:
			return self.netMain(tenInput) + tenInput

		elif self.netShortcut is not None:
			return self.netMain(tenInput) + self.netShortcut(tenInput)

		# end
	# end
# end

class Downsample3(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Upsample3(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Inpaint(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.netContext = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25)
		)

		self.netInput = Basic3('conv-relu-conv', [ 3 + 1 + 64 + 1, 32, 32 ])

		for intRow, intFeatures in [ (0, 32), (1, 64), (2, 128), (3, 256) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic3('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic3('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic3('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample3([ 32, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample3([ 64, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample3([ 128, 256, 256 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample3([ 256, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample3([ 128, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample3([ 64, 32, 32 ]))
		# end

		self.netImage = Basic3('conv-relu-conv', [ 32, 32, 3 ])
		self.netDisparity = Basic3('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tenImage, tenDisparity, tenShift):
		tenDepth = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDisparity + 0.0000001)
		tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
		tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
		tenPoints = tenPoints.view(1, 3, -1)

		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenContext = self.netContext(torch.cat([ tenImage, tenDisparity ], 1))

		tenRender, tenExisting = render_pointcloud(tenPoints + tenShift, torch.cat([ tenImage, tenDisparity, tenContext ], 1).view(1, 68, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

		tenExisting = (tenExisting > 0.0).float()
		tenExisting = tenExisting * spatial_filter(tenExisting, 'median-5')
		tenRender = tenRender * tenExisting.clone().detach()

		tenColumn = [ None, None, None, None ]

		tenColumn[0] = self.netInput(torch.cat([ tenRender, tenExisting ], 1))
		tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
		tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
		tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

		intColumn = 1
		for intRow in range(len(tenColumn)):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != 0:
				tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		tenImage = self.netImage(tenColumn[0])
		tenImage *= tenStd[0] + 0.0000001
		tenImage += tenMean[0]

		tenDisparity = self.netDisparity(tenColumn[0])
		tenDisparity *= tenStd[1] + 0.0000001
		tenDisparity += tenMean[1]

		return {
			'tenExisting': tenExisting,
			'tenImage': tenImage.clip(0.0, 1.0) if self.training == False else tenImage,
			'tenDisparity': torch.nn.functional.threshold(input=tenDisparity, threshold=0.0, value=0.0)
		}
	# end
# end

# inpaint
netInpaint = Inpaint()
netInpaint.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/kenburns/network-inpainting.pytorch', file_name='kenburns-inpainting').items() })


def get_size_in_MB(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    total_params = int(sum(p.numel() for p in model.parameters()))
    print('total_params:',total_params)
    return size_all_mb, total_params


dictModels = {0: "netInpaint", 1: "netRefine", 2: "netDisparity", 3: "netSemantics", 4: "netMaskrcnn"}
for i, e in enumerate([netInpaint, netRefine, netDisparity, netSemantics, netMaskrcnn]):
	print(f"Model: {dictModels[i]}")
	get_size_in_MB(e)
	# flops, params = flopth(e, 
	# 					inputs=(torch.rand(1, 3, 256, 192), 
	# 		  					torch.rand(1, 3, 256, 192), 
	# 							torch.rand(1, 3, 256, 192),)
	# 							)
	# print('flops:', flops, 'params:', params)
