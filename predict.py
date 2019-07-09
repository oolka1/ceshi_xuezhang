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
from Unet import UNet
from PIL import Image
from torchvision import transforms

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='./model_checkpoint/fudanc0_model_599.pth', metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
config = parser.parse_args()

testdata_root = "test"
save_root = "result"
test_dataset = fudandataset(testdata_root,train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, 
                                              num_workers=4)
num_classes = 4
classifier = UNet(n_classes = num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
classifier.load_state_dict(torch.load(config.model))

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
    
