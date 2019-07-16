import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
from utils import *
from models.net import SARPN
from options import get_args
from collections import OrderedDict
from dataloader import nyudv2_dataloader

args = get_args('test')
# lode nyud v2 test set
TestImgLoader = nyudv2_dataloader.getTestingData_NYUDV2(args.batch_size, args.testlist_path, args.root_path)
# model
model = SARPN(args)
model = nn.DataParallel(model)
model.cuda()

# load test model
if args.loadckpt is not None:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict)
else:
    print("You have not loaded any models.")

def test():
    model.eval()

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    for batch_idx, sample in enumerate(TestImgLoader):
        print("Processing the {}th image!".format(batch_idx))
        image, depth  = sample['image'], sample['depth']
        depth = depth.cuda()
        image = image.cuda()

        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        
        start = time.time()
        pred_depth = model(image)
        end = time.time()
        running_time = end - start
        output = torch.nn.functional.interpolate(pred_depth[-1], size=[depth.size(2), depth.size(3)], mode='bilinear', align_corners=True)

        depth_edge = edge_detection(depth)
        output_edge = edge_detection(output)
        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        errors = evaluateError(output, depth)
        errorSum = addErrors(errorSum, errors, batchSize)
        averageError = averageErrors(errorSum, totalNumber)

        edge1_valid = (depth_edge > args.threshold)
        edge2_valid = (output_edge > args.threshold)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / (depth.size(2)*depth.size(3))

        nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
        P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
        R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

        F = (2 * P * R) / (P + R)

        Ae += A
        Pe += P
        Re += R
        Fe += F

    Av = Ae / totalNumber
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print('PV', Pv)
    print('RV', Rv)
    print('FV', Fv)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)


if __name__ == '__main__':
    test()
