import os
import math
import torch
import numpy as np
import matplotlib
import matplotlib.cm
import torchvision.utils as vutils
from models.loss import Sobel

def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def adjust_learning_rate(optimizer, epoch, init_lr):

	lr = init_lr * (0.1 ** (epoch // 5))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, filename):
	torch.save(state, filename)


def edge_detection(depth):
	get_edge = Sobel().cuda()

	edge_xy = get_edge(depth)
	edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
		torch.pow(edge_xy[:, 1, :, :], 2)
	edge_sobel = torch.sqrt(edge_sobel)

	return edge_sobel

def build_optimizer(model,
					learning_rate, 
					optimizer_name='rmsprop',
					weight_decay=1e-5,
					epsilon=0.001,
					momentum=0.9):
	"""Build optimizer"""
	if optimizer_name == "sgd":
		print("Using SGD optimizer.")
		optimizer = torch.optim.SGD(model.parameters(), 
									lr = learning_rate,
									momentum=momentum,
									weight_decay=weight_decay)

	elif optimizer_name	== 'rmsprop':
		print("Using RMSProp optimizer.")
		optimizer = torch.optim.RMSprop(model.parameters(),
										lr = learning_rate,
										eps = epsilon,
										weight_decay = weight_decay,
										momentum = momentum
										)
	elif optimizer_name == 'adam':
		print("Using Adam optimizer.")
		optimizer = torch.optim.Adam(model.parameters(), 
									 lr = learning_rate, weight_decay=weight_decay)
	return optimizer




#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
	

def lg10(x):
	return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
	z = x.clone()
	maskYLarger = torch.lt(x, y)
	z[maskYLarger.detach()] = y[maskYLarger.detach()]
	return z

def nValid(x):
	return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
	return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
	return torch.ne(x, x)

def setNanToZero(input, target):
	nanMask = getNanMask(target)
	nValidElement = nValid(target)

	_input = input.clone()
	_target = target.clone()

	_input[nanMask] = 0
	_target[nanMask] = 0

	return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
	errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
			  'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	_output, _target, nanMask, nValidElement = setNanToZero(output, target)

	if (nValidElement.data.cpu().numpy() > 0):
		diffMatrix = torch.abs(_output - _target)

		errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

		errors['MAE'] = torch.sum(diffMatrix) / nValidElement

		realMatrix = torch.div(diffMatrix, _target)
		realMatrix[nanMask] = 0
		errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

		LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
		LG10Matrix[nanMask] = 0
		errors['LG10'] = torch.sum(LG10Matrix) / nValidElement
		yOverZ = torch.div(_output, _target)
		zOverY = torch.div(_target, _output)

		maxRatio = maxOfTwo(yOverZ, zOverY)

		errors['DELTA1'] = torch.sum(
			torch.le(maxRatio, 1.25).float()) / nValidElement
		errors['DELTA2'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
		errors['DELTA3'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

		errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
		errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
		errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
		errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
		errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
		errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
		errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

	return errors


def addErrors(errorSum, errors, batchSize):
	errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
	errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
	errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize
	errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

	errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
	errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
	errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

	return errorSum


def averageErrors(errorSum, N):
	averageError={'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
					'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	averageError['MSE'] = errorSum['MSE'] / N
	averageError['ABS_REL'] = errorSum['ABS_REL'] / N
	averageError['LG10'] = errorSum['LG10'] / N
	averageError['MAE'] = errorSum['MAE'] / N

	averageError['DELTA1'] = errorSum['DELTA1'] / N
	averageError['DELTA2'] = errorSum['DELTA2'] / N
	averageError['DELTA3'] = errorSum['DELTA3'] / N

	return averageError


def colormap(image, cmap="jet"):
	image_min = torch.min(image)
	image_max = torch.max(image)
	image = (image - image_min) / (image_max - image_min)
	image = torch.squeeze(image)

	# quantize 
	indices = torch.round(image * 255).long()
	# gather
	cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')

	colors = cm(np.arange(256))[:, :3]
	colors = torch.cuda.FloatTensor(colors)
	color_map = colors[indices].transpose(2, 3).transpose(1, 2)

	return color_map



def draw_losses(logger, loss, global_step):
	name = "train_loss"
	logger.add_scalar(name, loss, global_step)

def draw_images(logger, all_draw_image, global_step):
	for image_name, images in all_draw_image.items():
		if images.shape[1] == 1:
			images = colormap(images)
		elif images.shape[1] == 3:
			__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
						'std': [0.229, 0.224, 0.225]}
			for channel in np.arange(images.shape[1]):
				images[:, channel, :, :]  = images[:, channel, :, :] * __imagenet_stats["std"][channel] + __imagenet_stats["mean"][channel] 

		if len(images.shape) == 3:
			images = images[np.newaxis, :, :, :] 
		if images.shape[0]>4:
			images = images[:4, :, :, :]
		logger.add_image(image_name, images, global_step)






	
