import torch
import torch.nn as nn
import numpy as np
from models import modules
from models import get_models

class SARPN(nn.Module):
	def __init__(self, args):
		super(SARPN, self).__init__()
		print("backbone:", args.backbone)
		self.feature_extraction = get_models(args)

		if args.backbone in ["ResNet18", "ResNet34"]:
			adff_num_features = 640
			rpd_num_features = 512
			block_channel = [64, 64, 128, 256, 512]
			top_num_features = block_channel[-1]

		if args.backbone in ["ResNet50", "ResNet101", "ResNet152"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [64, 256, 512, 1024, 2048]
			top_num_features = block_channel[-1]

		if args.backbone in ["DenseNet121"]:
			adff_num_features = 640
			rpd_num_features = 1024
			block_channel = [64, 128, 256, 512, 1024]
			top_num_features = block_channel[-1]

		if args.backbone in ["DenseNet161"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [96, 192, 384, 1056, 2208]
			top_num_features = block_channel[-1]

		if args.backbone in ["DenseNet169"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [64, 128, 256, 640, 1664]
			top_num_features = block_channel[-1]

		if args.backbone in ["DenseNet201"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [64, 128, 256, 896, 1920]
			top_num_features = block_channel[-1]

		if args.backbone in ["SENet154"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [128, 256, 512, 1024, 2048]
			top_num_features = block_channel[-1]

		if args.backbone in ["SE_ResNet50", "SE_ResNet101", "SE_ResNet152", "SE_ResNext50_32x4d", "SE_ResNext101_32x4d"]:
			adff_num_features = 1280
			rpd_num_features = 2048
			block_channel = [64, 256, 512, 1024, 2048]
			top_num_features = block_channel[-1]

		self.residual_pyramid_decoder = modules.RPD(rpd_num_features, top_num_features)
		self.adaptive_dense_feature_fusion = modules.ADFF(block_channel, adff_num_features, rpd_num_features)

	def forward(self, x):
		feature_pyramid = self.feature_extraction(x)
		fused_feature_pyramid = self.adaptive_dense_feature_fusion(feature_pyramid)
		multiscale_depth = self.residual_pyramid_decoder(feature_pyramid, fused_feature_pyramid)

		return multiscale_depth
