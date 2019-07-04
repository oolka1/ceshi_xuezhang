#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:10:07 2019

@author: hesun
"""

import argparse
import torch
import torch.utils.data
import numpy as np
from fudandataset import fudandataset

from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='./model_checkpoint/fudanc0_model_399.pth', metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
config = parser.parse_args()

testdata_root = "test"
save_root = "result"
test_dataset = fudandataset(testdata_root,train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, 
                                              num_workers=4)
num_classes = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier=torch.load(config.model)
classifier.to(device)
test_acc_all = []
for j, data in enumerate(testdataloader):
    slices,label = data
    slices, label = slices.to(device), label.to(device)
    classifier = classifier.eval()
    pred = classifier(slices)
    
    indata = slices.cpu().numpy()
    inlabel = label.cpu().numpy()
    
    np.save('%s/raw_%d' % (save_root,j),indata)
    np.save('%s/label_%d' % (save_root,j),inlabel)
    pred = pred.view(-1, num_classes)
    label = label.view(-1).long()
    pred_choice = pred.data.max(1)[1]
    mask = pred_choice.detach().cpu().numpy()
    np.save('%s/mask_%d' % (save_root,j),mask)
    correct = pred_choice.eq(label.data).cpu().sum()
    test_acc = correct.item()/float(label.shape[0])
    print(('test image: %d | test acc: %f') 
                % (j+1, test_acc))
    test_acc_all.append(test_acc)
print(('mean test acc: %f') % (np.mean(test_acc_all)))
    
