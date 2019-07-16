import os
import torch
import torchvision
from models.modules import E_resnet, E_densenet, E_senet
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.densenet import densenet161, densenet121, densenet169, densenet201
from models.senet import senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
import pdb

__models__ = {
	'ResNet18': lambda :E_resnet(resnet18(pretrained = True)),
	'ResNet34': lambda :E_resnet(resnet34(pretrained = True)),
	'ResNet50': lambda :E_resnet(resnet50(pretrained = True)),
	'ResNet101': lambda :E_resnet(resnet101(pretrained = True)),
	'ResNet152': lambda :E_resnet(resnet152(pretrained = True)),
	'DenseNet121': lambda :E_densenet(densenet121(pretrained = True)),
	'DenseNet161': lambda :E_densenet(densenet161(pretrained = True)),
	'DenseNet169': lambda :E_densenet(densenet169(pretrained = True)),
	'DenseNet201': lambda :E_densenet(densenet201(pretrained = True)),
	'SENet154': lambda :E_senet(senet154(pretrained="imagenet")),
	'SE_ResNet50': lambda :E_senet(se_resnet50(pretrained="imagenet")),
	'SE_ResNet101': lambda :E_senet(se_resnet101(pretrained="imagenet")),
	'SE_ResNet152': lambda :E_senet(se_resnet152(pretrained="imagenet")),
	'SE_ResNext50_32x4d': lambda :E_senet(se_resnext50_32x4d(pretrained="imagenet")),
	'SE_ResNext101_32x4d': lambda :E_senet(se_resnext101_32x4d(pretrained="imagenet"))
}   


def get_models(args):
    backbone = args.backbone

    if os.getenv('TORCH_MODEL_ZOO') != args.pretrained_dir:
        os.environ['TORCH_MODEL_ZOO'] = args.pretrained_dir
    else:
        pass

    return __models__[backbone]()

