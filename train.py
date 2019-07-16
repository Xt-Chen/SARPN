import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
from options import get_args
from models.net import SARPN
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataloader import nyudv2_dataloader
from models.loss import adjust_gt, total_loss

cudnn.benchmark = True
args = get_args('train')

# Create folder
makedir(args.checkpoint_dir)
makedir(args.logdir)

# creat summary logger
logger = SummaryWriter(args.logdir)

# dataset, dataloader
TrainImgLoader = nyudv2_dataloader.getTrainingData_NYUDV2(args.batch_size, args.trainlist_path, args.root_path)
# model, optimizer
model = SARPN(args)
model = nn.DataParallel(model)
model.cuda()

optimizer = build_optimizer(model = model,
							learning_rate=args.lr,
							optimizer_name=args.optimizer_name,
							weight_decay = args.weight_decay,
							epsilon=args.epsilon,
							momentum=args.momentum
							)

# load parameters
start_epoch = 0
## progress
if args.resume:
	all_saved_ckpts = [ckpt for ckpt in os.listdir(args.checkpoint_dir) if ckpt.endswith(".pth.tar")]
	all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x:int(x.split('_')[-1].split('.'))[0])
	loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
	start_epoch = all_saved_ckpts[-1].split('_')[-1].split('.')[0]
	print("loading the lastest model in checkpoint_dir: {}".format(loadckpt))
	state_dict = torch.load(loadckpt)
	model.load_state_dict(state_dict)
elif args.loadckpt is not None:
	print("loading model {}".format(args.loadckpt))
	start_epoch = args.loadckpt.split('_')[-1].split('.')[0]
	state_dict = torch.load(args.loadckpt)
	model.load_state_dict(state_dict)
else:
	print("start at epoch {}".format(start_epoch))

## train process
def train():
	for epoch in range(start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args.lr)
		batch_time = AverageMeter()
		losses = AverageMeter()
		model.train()
		end = time.time()
		for batch_idx, sample in enumerate(TrainImgLoader):
			image, depth = sample['image'], sample['depth']           
			depth = depth.cuda()
			image = image.cuda()
			image = torch.autograd.Variable(image)
			depth = torch.autograd.Variable(depth)			
			optimizer.zero_grad()
			global_step = len(TrainImgLoader) * epoch + batch_idx
			# get the predicted depth maps of different scales
			pred_depth = model(image)
			# adjust ground-truth to the corresponding scales
			gt_depth = adjust_gt(depth, pred_depth)
			# Calculate the total loss 
			loss = total_loss(pred_depth, gt_depth)

			losses.update(loss.item(), image.size(0))
			loss.backward()
			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()
			
			if args.do_summary:
				all_draw_image = {"image":image, "pred":pred_depth[-1], "gt":gt_depth[-1]}
				draw_losses(logger, losses.avg, global_step)
				draw_images(logger, all_draw_image, global_step)
			
			#batchSize = depth.size(0)

			print(('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})'
			.format(epoch, batch_idx, len(TrainImgLoader), batch_time=batch_time, loss=losses)))

		if (epoch+1)%1 == 0:
			save_checkpoint(model.state_dict(), filename=args.checkpoint_dir + "SARPN_checkpoints_" + str(epoch + 1) + ".pth.tar")

if __name__ == '__main__':
	train()
