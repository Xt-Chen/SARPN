from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from models import senet
from models import resnet
from models import densenet
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)      
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,          
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,             
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear',align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class E_resnet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_block0 = x

        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]

        return feature_pyramid

class E_densenet(nn.Module):

    def __init__(self, original_model, num_features = 2208):
        super(E_densenet, self).__init__()        
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x_block0 = x03
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]

        return feature_pyramid

class E_senet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_senet, self).__init__()        
        self.base = nn.Sequential(*list(original_model.children())[:-3])
    
    def forward(self, x):
        x_block0 = nn.Sequential(*list(self.base[0].children())[:-1])(x)      
        x0 = self.base[0](x)       
        x_block1 = self.base[1](x0)                                           
        x_block2 = self.base[2](x_block1)                                     
        x_block3 = self.base[3](x_block2)                                     
        x_block4 = self.base[4](x_block3)                                    
        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]
        return feature_pyramid


class Refineblock(nn.Module):
    def __init__(self, num_features, kernel_size):
        super(Refineblock, self).__init__()
        padding=(kernel_size-1)//2

        self.conv1 = nn.Conv2d(1, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features//2)

        self.conv2 = nn.Conv2d(  
            num_features//2, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features//2)

        self.conv3 = nn.Conv2d(num_features//2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True)



    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.bn1(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv2(x_res)
        x_res = self.bn2(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv3(x_res)

        x2 = x  + x_res
        return x2

# Residual Pyramid Decoder
class RPD(nn.Module):

    def __init__(self, rpd_num_features = 2048, top_num_features=2048):
        super(RPD, self).__init__()


        self.conv = nn.Conv2d(top_num_features, rpd_num_features // 2, kernel_size=1, stride=1, bias=False)                                               
        self.bn = nn.BatchNorm2d(rpd_num_features//2)                                                    

        self.conv5 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features//2, kernel_size=3, stride=1, padding=1, bias=False),    
                                   nn.BatchNorm2d(rpd_num_features//2),                                                             
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features//2, 1, kernel_size=3, stride=1, padding=1, bias=False))               
        rpd_num_features = rpd_num_features // 2                                                                                              
        self.scale5 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                       

        self.conv4 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale4 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        
        

        self.conv3 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale3 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv2 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.scale2 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv1 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale1 = Refineblock(num_features=rpd_num_features, kernel_size=3)
    def forward(self, feature_pyramid, fused_feature_pyramid):

        scale1_size = [fused_feature_pyramid[0].size(2), fused_feature_pyramid[0].size(3)]
        scale2_size = [fused_feature_pyramid[1].size(2), fused_feature_pyramid[1].size(3)]
        scale3_size = [fused_feature_pyramid[2].size(2), fused_feature_pyramid[2].size(3)]
        scale4_size = [fused_feature_pyramid[3].size(2), fused_feature_pyramid[3].size(3)]
        scale5_size = [fused_feature_pyramid[4].size(2), fused_feature_pyramid[4].size(3)]
        

        # scale5
        scale5 = torch.cat((F.relu(self.bn(self.conv(feature_pyramid[4]))), fused_feature_pyramid[4]), 1)
        scale5_depth = self.scale5(self.conv5(scale5))

        # scale4
        scale4_res = self.conv4(fused_feature_pyramid[3])
        scale5_upx2 = F.interpolate(scale5_depth, size=scale4_size,
                                    mode='bilinear', align_corners=True)
        scale4_depth = self.scale4(scale4_res + scale5_upx2)

        # scale3 
        scale3_res = self.conv3(fused_feature_pyramid[2])
        scale4_upx2 = F.interpolate(scale4_depth, size=scale3_size,
                                    mode='bilinear', align_corners=True)
        scale3_depth = self.scale3(scale3_res + scale4_upx2)

        # scale2
        scale2_res = self.conv2(fused_feature_pyramid[1])
        scale3_upx2 = F.interpolate(scale3_depth, size=scale2_size,
                                    mode='bilinear', align_corners=True)
        scale2_depth = self.scale2(scale2_res + scale3_upx2)

        # scale1
        scale1_res = self.conv1(fused_feature_pyramid[0])
        scale2_upx2 = F.interpolate(scale2_depth, size=scale1_size,
                                    mode='bilinear', align_corners=True)
        scale1_depth = self.scale1(scale1_res + scale2_upx2)

        scale_depth = [scale5_depth, scale4_depth, scale3_depth, scale2_depth, scale1_depth]

        return scale_depth



# Adaptive Dense Features Fusion module
class ADFF(nn.Module):

    def __init__(self, block_channel, adff_num_features=1280, rpd_num_features=2048):
        super(ADFF, self).__init__()

        rpd_num_features = rpd_num_features // 2
        print("block_channel:", block_channel)
        #scale5
        self.upsample_scale1to5 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5)  
        self.upsample_scale2to5 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5)  
        self.upsample_scale3to5 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5)  
        self.upsample_scale4to5 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5)  
        self.upsample_scale5to5 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5)  
        self.conv_scale5 = nn.Conv2d(adff_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 1280/1024
        self.bn_scale5 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2                
        rpd_num_features = rpd_num_features // 2           

        # scale4
        self.upsample_scale1to4 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5) 
        self.upsample_scale2to4 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5) 
        self.upsample_scale3to4 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5) 
        self.upsample_scale4to4 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5) 
        self.upsample_scale5to4 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5) 
        self.conv_scale4 = nn.Conv2d(adff_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 640/512
        self.bn_scale4 = nn.BatchNorm2d(rpd_num_features) 

        adff_num_features = adff_num_features // 2                 
        rpd_num_features = rpd_num_features // 2           

        # scale3
        self.upsample_scale1to3 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5)  
        self.upsample_scale2to3 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5)  
        self.upsample_scale3to3 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5)  
        self.upsample_scale4to3 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5)  
        self.upsample_scale5to3 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5)  
        self.conv_scale3 = nn.Conv2d(adff_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 320/256
        self.bn_scale3 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2                  
        rpd_num_features = rpd_num_features // 2

        # scale2
        self.upsample_scale1to2 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5)  
        self.upsample_scale2to2 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5)  
        self.upsample_scale3to2 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5)  
        self.upsample_scale4to2 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5)  
        self.upsample_scale5to2 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5)  
        self.conv_scale2 = nn.Conv2d(adff_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)        # 160/128
        self.bn_scale2 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2                  
        rpd_num_features = rpd_num_features // 2                    

        #scale1   
        self.upsample_scale1to1 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5)  
        self.upsample_scale2to1 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5)  
        self.upsample_scale3to1 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5)  
        self.upsample_scale4to1 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5) 
        self.upsample_scale5to1 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5)  
        self.conv_scale1 = nn.Conv2d(adff_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)        # 80/64
        self.bn_scale1 = nn.BatchNorm2d(rpd_num_features)
    
    def forward(self, feature_pyramid):
        scale1_size = [feature_pyramid[0].size(2), feature_pyramid[0].size(3)]
        scale2_size = [feature_pyramid[1].size(2), feature_pyramid[1].size(3)]
        scale3_size = [feature_pyramid[2].size(2), feature_pyramid[2].size(3)]
        scale4_size = [feature_pyramid[3].size(2), feature_pyramid[3].size(3)]
        scale5_size = [feature_pyramid[4].size(2), feature_pyramid[4].size(3)]


        # scale5_mff       8x10
        scale_1to5 = self.upsample_scale1to5(feature_pyramid[0], scale5_size)
        scale_2to5 = self.upsample_scale2to5(feature_pyramid[1], scale5_size)
        scale_3to5 = self.upsample_scale3to5(feature_pyramid[2], scale5_size)
        scale_4to5 = self.upsample_scale4to5(feature_pyramid[3], scale5_size)
        scale_5to5 = self.upsample_scale5to5(feature_pyramid[4], scale5_size)
        scale5_mff = torch.cat((scale_1to5, scale_2to5, scale_3to5, scale_4to5, scale_5to5), 1)
        scale5_mff = F.relu(self.bn_scale5(self.conv_scale5(scale5_mff)))                         

        # scale4_mff       15x19
        scale_1to4 = self.upsample_scale1to4(feature_pyramid[0], scale4_size)
        scale_2to4 = self.upsample_scale2to4(feature_pyramid[1], scale4_size)
        scale_3to4 = self.upsample_scale3to4(feature_pyramid[2], scale4_size)
        scale_4to4 = self.upsample_scale4to4(feature_pyramid[3], scale4_size)
        scale_5to4 = self.upsample_scale5to4(feature_pyramid[4], scale4_size)
        scale4_mff = torch.cat((scale_1to4, scale_2to4, scale_3to4, scale_4to4, scale_5to4), 1)
        scale4_mff = F.relu(self.bn_scale4(self.conv_scale4(scale4_mff)))                        

        # scale3_mff       29x38
        scale_1to3 = self.upsample_scale1to3(feature_pyramid[0], scale3_size)
        scale_2to3 = self.upsample_scale2to3(feature_pyramid[1], scale3_size)
        scale_3to3 = self.upsample_scale3to3(feature_pyramid[2], scale3_size)
        scale_4to3 = self.upsample_scale4to3(feature_pyramid[3], scale3_size)
        scale_5to3 = self.upsample_scale5to3(feature_pyramid[4], scale3_size)
        scale3_mff = torch.cat((scale_1to3, scale_2to3, scale_3to3, scale_4to3, scale_5to3), 1)
        scale3_mff = F.relu(self.bn_scale3(self.conv_scale3(scale3_mff)))                        

        # scale2_mff      57x76
        scale_1to2 = self.upsample_scale1to2(feature_pyramid[0], scale2_size)
        scale_2to2 = self.upsample_scale2to2(feature_pyramid[1], scale2_size)
        scale_3to2 = self.upsample_scale3to2(feature_pyramid[2], scale2_size)
        scale_4to2 = self.upsample_scale4to2(feature_pyramid[3], scale2_size)
        scale_5to2 = self.upsample_scale5to2(feature_pyramid[4], scale2_size)
        scale2_mff = torch.cat((scale_1to2, scale_2to2, scale_3to2, scale_4to2, scale_5to2), 1)
        scale2_mff = F.relu(self.bn_scale2(self.conv_scale2(scale2_mff)))                        

        # scale1_mff      114x152
        scale_1to1 = self.upsample_scale1to1(feature_pyramid[0], scale1_size)             
        scale_2to1 = self.upsample_scale2to1(feature_pyramid[1], scale1_size)
        scale_3to1 = self.upsample_scale3to1(feature_pyramid[2], scale1_size)
        scale_4to1 = self.upsample_scale4to1(feature_pyramid[3], scale1_size)
        scale_5to1 = self.upsample_scale5to1(feature_pyramid[4], scale1_size)
        scale1_mff = torch.cat((scale_1to1, scale_2to1, scale_3to1, scale_4to1, scale_5to1), 1)
        scale1_mff = F.relu(self.bn_scale1(self.conv_scale1(scale1_mff)))                           

        fused_feature_pyramid = [scale1_mff, scale2_mff, scale3_mff, scale4_mff, scale5_mff]

        return fused_feature_pyramid
