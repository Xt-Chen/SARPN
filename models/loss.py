import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def adjust_gt(gt_depth, pred_depth):
	adjusted_gt = []
	for each_depth in pred_depth:
		adjusted_gt.append(F.interpolate(gt_depth, size=[each_depth.size(2), each_depth.size(3)],
								   mode='bilinear', align_corners=True))
	return adjusted_gt

class Sobel(nn.Module):
	def __init__(self):
		super(Sobel, self).__init__()
		self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
		edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
		edge_k = np.stack((edge_kx, edge_ky))

		edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
		self.edge_conv.weight = nn.Parameter(edge_k)
		
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		out = self.edge_conv(x) 
		out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
		return out

def total_loss(output, depth_gt):

	losses=[]

	for depth_index in range(len(output)):

		cos = nn.CosineSimilarity(dim=1, eps=0)
		get_gradient = Sobel().cuda()
		ones = torch.ones(depth_gt[depth_index].size(0), 1, depth_gt[depth_index].size(2),depth_gt[depth_index].size(3)).float().cuda()
		ones = torch.autograd.Variable(ones)
		depth_grad = get_gradient(depth_gt[depth_index])
		output_grad = get_gradient(output[depth_index])
		depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth_gt[depth_index])
		depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth_gt[depth_index])
		output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_gt[depth_index])
		output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_gt[depth_index])

		depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
		output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

		loss_depth = torch.log(torch.abs(output[depth_index] - depth_gt[depth_index]) + 0.5).mean()
		loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
		loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
		loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

		loss = loss_depth + loss_normal + (loss_dx + loss_dy)

		losses.append(loss)


	total_loss = sum(losses)
	
	return total_loss

